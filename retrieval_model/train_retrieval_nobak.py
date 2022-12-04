import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import os
import pdb
import wandb

from args_retrieval import get_parser
from datasets_retrieval import Dataset, train_transform
from models_retrieval_nobak import TextEncoder, ImageEncoder
from triplet_loss import global_loss, TripletLoss
from modules import DynamicSoftMarginLoss
import sys
sys.path.append('../')
from common import param_counter, sample_data

def create_model(ckpt_args, device='cuda'):
    text_encoder = TextEncoder(
        emb_dim=ckpt_args.word2vec_dim, 
        hid_dim=ckpt_args.rnn_hid_dim, 
        z_dim=ckpt_args.feature_dim, 
        data_dir=ckpt_args.data_dir, 
        text_info=ckpt_args.text_info,
        with_attention=ckpt_args.with_attention,
        ingr_enc_type=ckpt_args.ingrs_enc_type)
    image_encoder = ImageEncoder(
        z_dim=ckpt_args.feature_dim)
    text_encoder, image_encoder = [x.to(device) for x in [text_encoder, image_encoder]]
    print('# text_encoder', param_counter(text_encoder.parameters()))
    print('# image_encoder', param_counter(image_encoder.parameters()))
    if device == 'cuda':
        text_encoder, image_encoder = [nn.DataParallel(x) for x in [text_encoder, image_encoder]]
    optimizer = torch.optim.Adam([
            {'params': text_encoder.parameters()},
            {'params': image_encoder.parameters()},
        ], lr=ckpt_args.lr, betas=(0.5, 0.999))
    return text_encoder, image_encoder, optimizer

def load_model(ckpt_path, device='cuda'):
    print('load retrieval model from:', ckpt_path)
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']
    batch_idx = ckpt['batch_idx']
    text_encoder, image_encoder, optimizer = create_model(ckpt_args, device)
    if device=='cpu':
        text_encoder.load_state_dict(ckpt['text_encoder'])
        image_encoder.load_state_dict(ckpt['image_encoder'])
    else:
        text_encoder.module.load_state_dict(ckpt['text_encoder'])
        image_encoder.module.load_state_dict(ckpt['image_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt_args, batch_idx, text_encoder, image_encoder, optimizer


def save_model(args, batch_idx, text_encoder, image_encoder, optimizer, ckpt_path):
    print('save retrieval model to:', ckpt_path)
    ckpt = {
        'args': args,
        'batch_idx': batch_idx,
        'text_encoder': text_encoder.state_dict(),
        'image_encoder': image_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)

# hinge loss
def compute_loss(txt_feat, img_feat, device='cuda'):
    BS = txt_feat.shape[0]
    denom = img_feat.norm(p=2, dim=1, keepdim=True) @ txt_feat.norm(p=2, dim=1, keepdim=True).t()
    numer = img_feat @ txt_feat.t()
    sim = numer / (denom + 1e-12)
    margin = 0.3 * torch.ones_like(sim)
    mask = torch.eye(margin.shape[0], margin.shape[1]).bool().to(device)
    margin.masked_fill_(mask, 0)
    pos_sim = (torch.diag(sim) * torch.ones(BS, BS).to(device)).t() # [BS, BS]
    loss_retrieve_txt = torch.max(
        torch.tensor(0.0).to(device), 
        margin + sim - pos_sim)
    loss_retrieve_img = torch.max(
        torch.tensor(0.0).to(device), 
        margin + sim.t() - pos_sim)
    loss = loss_retrieve_img + loss_retrieve_txt
    # effective number of pairs is BS*BS-BS, those on the diagnal are never counted and always zero
    loss = loss.sum() / (BS*BS-BS) / 2.0
    return loss

def train(args, start_batch_idx, text_encoder, image_encoder, optimizer, train_loader, device='cuda'):    
    if args.loss_type == 'hinge':
        criterion = compute_loss
    elif args.loss_type == 'hardmining+hinge':
        triplet_loss = TripletLoss(margin=args.margin)
    elif args.loss_type == 'dynamic_soft_margin':
        criterion = DynamicSoftMarginLoss(is_binary=False, nbins=args.batch_size // 2)
        criterion = criterion.to(device)
    
    #####################
    # train
    #####################
    wandb.init(project="cookgan_retrieval_model")
    wandb.config.update(args)

    pbar = range(args.batches)
    pbar = tqdm(pbar, initial=start_batch_idx, dynamic_ncols=True, smoothing=0.3)

    text_encoder.train()
    image_encoder.train()
    if device=='cuda':
        text_module = text_encoder.module
        image_module = image_encoder.module
    else:
        text_module = text_encoder
        image_module = image_encoder
    train_loader = sample_data(train_loader)

    for batch_idx in pbar:
        txt, img = next(train_loader)
        for i in range(len(txt)):
            txt[i] = txt[i].to(device)
        img = img.to(device)

        txt_feat, _ = text_encoder(*txt)
        img_feat = image_encoder(img)
        bs = img.shape[0]
        if args.loss_type == 'hinge':
            loss = criterion(img_feat, txt_feat, device)
        elif args.loss_type == 'hardmining+hinge':
            label = list(range(0, bs))
            label.extend(label)
            label = np.array(label)
            label = torch.tensor(label).long().to(device)
            loss = global_loss(triplet_loss, torch.cat((img_feat, txt_feat)), label, normalize_feature=True)[0]
        elif args.loss_type == 'dynamic_soft_margin':
            out = torch.cat((img_feat, txt_feat))
            loss = criterion(out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({
            'training loss': loss,
            'batch_idx': batch_idx
        })

        if batch_idx % 10_000 == 0:
            ckpt_path = f'{wandb.run.dir}/{batch_idx:>08d}.ckpt'
            save_model(args, batch_idx, text_module, image_module, optimizer, ckpt_path)

if __name__ == '__main__':
    ##############################
    # setup
    ##############################
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device

    ##############################
    # dataset
    ##############################
    print('loading datasets')
    train_set = Dataset(
        part='train', 
        recipe_file=args.recipe_file,
        img_dir=args.img_dir, 
        word2vec_file=args.word2vec_file, 
        transform=train_transform, 
        permute_ingrs=args.permute_ingrs)

    if args.debug:
        print('in debug mode')
        train_set = torch.utils.data.Subset(train_set, range(2000))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    print('train data:', len(train_set), len(train_loader))

    ##########################
    # model
    ##########################
    if args.ckpt_path:
        ckpt_args, batch_idx, text_encoder, image_encoder, optimizer = load_model(args.ckpt_path, device)
    else:
        text_encoder, image_encoder, optimizer = create_model(args, device)
        batch_idx = 0

    train(args, batch_idx, text_encoder, image_encoder, optimizer, train_loader, device='cuda')
