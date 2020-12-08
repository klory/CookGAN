import torch
# from torch.utils.tensorboard import SummaryWriter
import wandb
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torch import optim
import numpy as np
import random
import pprint
import os
from tqdm import tqdm
import json
import pdb

from args_cookgan import get_parser
from datasets_cookgan import FoodDataset
from models_cookgan import G_NET, D_NET64, D_NET128, D_NET256
from utils_cookgan import prepare_data, compute_img_feat, compute_txt_feat, save_img_results

import sys
sys.path.append('../')
from common import param_counter, sample_data, clean_state_dict, requires_grad
sys.path.append('../retrieval_model')
import train_retrieval
import utils_retrieval

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def save_model(args, batch_idx, netG, optimizerG, netsD, optimizersD, ckpt_path):
    print('save to:', ckpt_path)
    ckpt = {}
    ckpt['args'] = args
    ckpt['batch_idx'] = batch_idx
    ckpt['netG'] = netG.state_dict()
    ckpt['optimizerG'] = optimizerG.state_dict()
    for i in range(len(netsD)):
        netD = netsD[i]
        optimizerD = optimizersD[i]
        ckpt['netD_{}'.format(i)] = netD.state_dict()
        ckpt['optimizerD_{}'.format(i)] = optimizerD.state_dict()
    torch.save(ckpt, ckpt_path)


def create_model(args, device='cuda'):
    netG = G_NET(
        gf_dim=64, z_dim=args.z_dim, 
        text_dim=args.input_dim, embedding_dim=args.embedding_dim, 
        r_num=2, levels=args.levels, b_condition=True, ca=True).to(device)
    netG.apply(weights_init)
    print('# params in netG =', param_counter(netG.parameters()))

    netsD = []
    netsD.append(D_NET64())
    if args.levels >= 2:
        netsD.append(D_NET128())
    if args.levels >= 3:
        netsD.append(D_NET256())
    for i in range(len(netsD)):
        netsD[i] = netsD[i].to(device)
        netsD[i].apply(weights_init)
        print('# params in netD_{} ='.format(i), param_counter(netsD[i].parameters()))
    
    if device=='cuda':
        netG = nn.DataParallel(netG)
        for i in range(len(netsD)):
            netsD[i] = nn.DataParallel(netsD[i])

    optimizerG = optim.Adam(netG.parameters(),
                            lr=args.lr_g,
                            betas=(0.5, 0.999))
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                            lr=args.lr_d,
                            betas=(0.5, 0.999))
        optimizersD.append(opt)
    return netG, netsD, optimizerG, optimizersD

def load_model(ckpt_path, device='cuda'):
    print('load CookGAN model from:', ckpt_path)
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']
    batch = ckpt['batch_idx']
    
    netG, netsD, optimizerG, optimizersD = create_model(ckpt_args, device)
    if device=='cuda':
        netG.module.load_state_dict(ckpt['netG'])
    else:
        netG.load_state_dict(ckpt['netG'])
    optimizerG.load_state_dict(ckpt['optimizerG'])
    for i in range(len(netsD)):
        if device=='cuda':
            netsD[i].module.load_state_dict(ckpt['netD_{}'.format(i)])
        else:
            netsD[i].load_state_dict(ckpt['netD_{}'.format(i)])
        optimizersD[i].load_state_dict(ckpt['optimizerD_{}'.format(i)])
    return ckpt_args, batch, netG, optimizerG, netsD, optimizersD

def compute_cycle_loss(feat1, feat2, paired=True, device='cuda'):
    if paired:
        loss = nn.CosineEmbeddingLoss(0.3)(feat1, feat2, torch.ones(feat1.shape[0]).to(device))
    else:
        loss = nn.CosineEmbeddingLoss(0.3)(feat1, feat2, -torch.ones(feat1.shape[0]).to(device))
    return loss

def compute_kl(mu, logvar, embedding_dim=128):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # return KLD.mean() # correct
    return KLD.mean() / embedding_dim # not correct, this is just to follow the official code


def train(
    args, batch_start, train_loader, device,
    txt_encoder, img_encoder, 
    netG, optimizerG, netsD, optimizersD, criterion,
    fixed_noise, fixed_txt, fixed_img, fixed_title, save_dir):

    if device=='cuda':
        netG_to_save = netG.module
        netsD_to_save = []
        for i in range(len(netsD)):
            netsD_to_save.append(netsD[i].module)
    else:
        netG_to_save = netG
        netsD_to_save = []
        for i in range(len(netsD)):
            netsD_to_save.append(netsD[i])
    
    noise = torch.FloatTensor(args.batch_size, args.z_dim).to(device)
    loader = sample_data(train_loader)
    for batch_idx in tqdm(range(batch_start, batch_start+args.num_batches)):
        if args.labels == 'original':
            real_labels = torch.FloatTensor(args.batch_size).fill_(
                1)  # (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float() #
            fake_labels = torch.FloatTensor(args.batch_size).fill_(
                0)  # (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() #
        elif args.labels == 'R-smooth':
            real_labels = torch.FloatTensor(args.batch_size).fill_(1) - (
                        torch.FloatTensor(args.batch_size).uniform_() * 0.1)
            fake_labels = (torch.FloatTensor(args.batch_size).uniform_() * 0.1)
        elif args.labels == 'R-flip':
            real_labels = (torch.FloatTensor(args.batch_size).uniform_() < 0.9).float()  #
            fake_labels = (torch.FloatTensor(args.batch_size).uniform_() > 0.9).float()  #
        elif args.labels == 'R-flip-smooth':
            real_labels = torch.abs((torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() - (
                    torch.FloatTensor(args.batch_size).fill_(1) - (
                        torch.FloatTensor(args.batch_size).uniform_() * 0.1)))
            fake_labels = torch.abs((torch.FloatTensor(args.batch_size).uniform_() > 0.9).float() - (
                    torch.FloatTensor(args.batch_size).uniform_() * 0.1))

        real_labels = real_labels.to(device)
        fake_labels = fake_labels.to(device)

        data = next(loader)
        txt, real_imgs, wrong_imgs = prepare_data(data, device)
        with torch.no_grad():
            txt_feat = compute_txt_feat(txt, txt_encoder)
        noise.normal_(0, 1).to(device)
        fake_imgs, mu, logvar = netG(noise, txt_feat)
        
        ######################
        # train Discriminators
        ######################
        errD_total = 0
        for level in range(args.levels):
            if args.input_noise:
                sigma = np.clip(1.0 - batch_idx/80_000, 0, 1) * 0.1
                real_img_noise = torch.empty_like(real_imgs[level]).normal_(0, sigma)
                wrong_img_noise = torch.empty_like(wrong_imgs[level]).normal_(0, sigma)
                fake_img_noise = torch.empty_like(fake_imgs[level]).normal_(0, sigma)
            else:
                real_img_noise = torch.zeros_like(real_imgs[level])
                wrong_img_noise = torch.zeros_like(wrong_imgs[level])
                fake_img_noise = torch.zeros_like(fake_imgs[level])

            netD = netsD[level]
            optD = optimizersD[level]
            real_logits = netD(real_imgs[level]+real_img_noise, mu.detach())
            wrong_logits = netD(wrong_imgs[level]+wrong_img_noise, mu.detach())
            fake_logits = netD(fake_imgs[level].detach()+fake_img_noise, mu.detach())

            errD_real = criterion(real_logits[0], real_labels) # cond_real --> 1
            errD_wrong = criterion(wrong_logits[0], fake_labels) # cond_wrong --> 0
            errD_fake = criterion(fake_logits[0], fake_labels) # cond_fake --> 0
            errD_cond = errD_real + errD_wrong + errD_fake
            
            if len(real_logits)>1:
                errD_real_uncond = criterion(real_logits[1], real_labels) # uncond_real --> 1
                errD_wrong_uncond = criterion(wrong_logits[1], real_labels) # uncond_wrong --> 1
                errD_fake_uncond = criterion(fake_logits[1], fake_labels) # uncond_fake --> 0
                errD_uncond = errD_real_uncond + errD_wrong_uncond + errD_fake_uncond
            else: # back to GAN-INT-CLS
                errD_cond = errD_real + 0.5 * (errD_wrong + errD_fake)
                errD_uncond = 0.0
            
            errD = errD_cond + args.uncond * errD_uncond
            
            optD.zero_grad()
            errD.backward()
            optD.step()

            # record
            errD_total += errD

            wandb.log({
                f'errD_cond{level}': errD_cond,
                f'errD_uncond{level}': errD_uncond,
                f'errD{level}': errD,
                f'batch_idx': batch_idx, 
            })
        
        ######################
        # train Generator
        ######################
        errG_total = 0.0
        for level in range(args.levels):
            if args.input_noise:
                sigma = np.clip(1.0 - batch_idx/80_000, 0, 1) * 0.1
                fake_img_noise = torch.empty_like(fake_imgs[level]).normal_(0, sigma)
            else:
                fake_img_noise = torch.zeros_like(fake_imgs[level])

            outputs = netsD[level](fake_imgs[level] + fake_img_noise, mu)
            errG_cond = criterion(outputs[0], real_labels) # cond_fake --> 1
            errG_uncond = criterion(outputs[1], real_labels) # uncond_fake --> 1

            fake_img_feat = compute_img_feat(fake_imgs[level], img_encoder)
            errG_cycle_txt = compute_cycle_loss(fake_img_feat, txt_feat)
            
            real_img_feat = compute_img_feat(real_imgs[level], img_encoder)
            errG_cycle_img = compute_cycle_loss(fake_img_feat, real_img_feat)

            # rightRcp_vs_rightImg = compute_cycle_loss(txt_feat, real_img_feat)
            # wrong_img_feat = compute_img_feat(wrong_imgs[level], img_encoder)
            # rightRcp_vs_wrongImg = compute_cycle_loss(txt_feat, wrong_img_feat, paired=False)
            # tri_loss = rightRcp_vs_rightImg + rightRcp_vs_wrongImg
            
            errG = errG_cond \
                + args.uncond * errG_uncond \
                    + args.cycle_txt * errG_cycle_txt \
                        + args.cycle_img * errG_cycle_img \
                            # + args.tri_loss * tri_loss

            # record
            errG_total += errG

            wandb.log({
                f'errG_cond{level}': errG_cond,
                f'errG_uncond{level}': errG_uncond,
                f'errG_cycle_txt{level}': errG_cycle_txt,
                f'errG_cycle_img{level}': errG_cycle_img,
                f'errG{level}': errG,
                f'batch_idx': batch_idx, 
            })
        
        errG_kl = compute_kl(mu, logvar)
        errG_total += args.kl * errG_kl

        optimizerG.zero_grad()
        errG_total.backward()
        optimizerG.step()
            

        wandb.log({
            f'errG_kl': errG_kl,
            f'errD_total': errD_total,
            f'errG_total': errG_total,
            f'batch_idx': batch_idx, 
        })

        if batch_idx % 1000 == 0:
            netG.eval()
            # save img and ckpt
            with torch.no_grad():
                fixed_txt_feat = compute_txt_feat(fixed_txt, txt_encoder)
                fake_imgs, mu, logvar = netG(fixed_noise, fixed_txt_feat)
                fake_img = fake_imgs[-1]
                real_fake = torch.stack([fixed_img.detach().cpu(), fake_img.detach().cpu()]).permute(1,0,2,3,4).contiguous()
                output_img = []
                for item, title in zip(real_fake, fixed_title):
                    item = vutils.make_grid(item, normalize=True, scale_each=True)
                    output_img.append(wandb.Image(item, caption=title[:100]))

            wandb.log({
                f'img': output_img,
                f'batch_idx': batch_idx, 
            })
            netG.train()
        if batch_idx % 10000 == 0:
            ckpt_path = os.path.join(save_dir, f'{batch_idx:>06d}.ckpt')
            save_model(args, batch_idx, netG_to_save, optimizerG, netsD_to_save, optimizersD, ckpt_path)
        
        batch_idx += 1


if __name__ == '__main__':
    args = get_parser().parse_args()
    ##############################
    # setup
    ##############################
    if not args.seed:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = args.device
    if device.__str__() == 'cpu':
        args.batch_size = 16

    ##############################
    # dataset
    ##############################
    imsize = args.base_size * (2 ** (args.levels-1))
    train_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    train_set = FoodDataset(
        recipe_file=args.recipe_file,
        img_dir=args.img_dir,
        levels=args.levels,
        part='train', 
        food_type=args.food_type,
        base_size=args.base_size, 
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size,
            drop_last=True, shuffle=True, num_workers=int(args.workers))
    print('train data info:', len(train_set), len(train_loader))

    ##############################
    # model
    ##############################
    ckpt_args, _, txt_encoder, img_encoder, _ = train_retrieval.load_model(args.retrieval_model, device)
    requires_grad(txt_encoder, False)
    requires_grad(img_encoder, False)
    txt_encoder = txt_encoder.eval()
    img_encoder = img_encoder.eval()

    if args.ckpt_path:
        ckpt_args, batch, netG, optimizerG, netsD, optimizersD = load_model(args.ckpt_path, device)
        wandb_run_id = args.ckpt_path.split('/')[-2]
        batch_start = batch + 1
    else:
        netG, netsD, optimizerG, optimizersD = create_model(args, device)
        wandb_run_id = ''
        batch_start = 0

    ##############################
    # train
    ##############################
    # define loss
    criterion = nn.BCELoss()

    # define fixed noise
    fixed_noise_part1 = torch.FloatTensor(1, args.z_dim).normal_(0, 1)
    fixed_noise_part1 = fixed_noise_part1.repeat(args.batch_size//2, 1)
    fixed_noise_part2 = torch.FloatTensor(args.batch_size//2, args.z_dim).normal_(0, 1)
    fixed_noise = torch.cat([fixed_noise_part1, fixed_noise_part2], dim=0).to(device)

    fixed_txt, fixed_imgs, _, fixed_title = next(iter(train_loader))
    fixed_img = fixed_imgs[-1]

    # setup saving directory
    project_name = "cookgan"
    wandb.init(project=project_name, config=args, resume=wandb_run_id)
    wandb.config.update(args)

    train(
        args, batch_start, train_loader, device,
        txt_encoder, img_encoder, 
        netG, optimizerG, netsD, optimizersD, criterion,
        fixed_noise, fixed_txt, fixed_img, fixed_title, wandb.run.dir)
