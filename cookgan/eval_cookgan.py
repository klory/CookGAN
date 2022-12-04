import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from args_cookgan import get_parser
from datasets_cookgan import FoodDataset
import sys
sys.path.append('/data/CS470_HnC/')
from common import load_retrieval_model, load_generation_model, mean, std, rank
from scipy.spatial.distance import cdist, pdist
import pdb
from types import SimpleNamespace

args = get_parser.parse_args()

assert args.resume != ''
args.batch_size = 64
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' \
    if torch.cuda.is_available() and args.cuda
    else 'cpu')
print('device:', device)

netG = load_generation_model(args.resume, device)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir('../')
TxtEnc, ImgEnc = load_retrieval_model(args.retrieval_model, device)
os.chdir(dname)

imsize = 256
image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imsize)])
dataset = FoodDataset(
    args.data_dir, args.img_dir, food_type=args.food_type, 
    levels=args.levels, part='test', 
    base_size=args.base_size, transform=image_transform)
# dataset = torch.utils.data.Subset(dataset, range(500))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size,
    drop_last=True, shuffle=False, num_workers=int(args.workers))
print('=> dataset dataloader =', len(dataset), len(dataloader))

generation_model_name = args.resume.rsplit('/', 1)[-1].rsplit('.', 1)[0]
save_dir = 'experiments/{}'.format(generation_model_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def prepare_data(data):
    imgs, w_imgs, txt, _ = data

    real_vimgs, wrong_vimgs = [], []
    for i in range(args.levels):
        real_vimgs.append(imgs[i].to(device))
        wrong_vimgs.append(w_imgs[i].to(device))
    
    vtxt = [x.to(device) for x in txt]
    return real_vimgs, wrong_vimgs, vtxt

# fixed_noise = torch.zeros(args.batch_size, args.z_dim).to(device)
fixed_noise = torch.FloatTensor(1, args.z_dim).normal_(0, 1).to(device)
fixed_noise = fixed_noise.repeat(args.batch_size, 1)
batch = 0

txt_feats_real = []
img_feats_real = []
img_feats_fake = []

def _get_img_embeddings(img, ImgEnc):
    img = img/2 + 0.5
    img = F.interpolate(img, [224, 224], mode='bilinear', align_corners=True)
    for i in range(img.shape[1]):
        img[:,i] = (img[:,i]-mean[i])/std[i]
    with torch.no_grad():
        img_feats = ImgEnc(img).detach().cpu()
    return img_feats

for data in tqdm(dataloader):
    real_imgs, _, txt = prepare_data(data)
    txt_embedding = TxtEnc(txt)
    with torch.no_grad():
        fake_imgs, _, _ = netG(fixed_noise, txt_embedding)
        
        txt_feats_real.append(txt_embedding.detach().cpu())
        img_fake = fake_imgs[-1]
        img_embedding_fake = _get_img_embeddings(img_fake, ImgEnc)
        img_feats_fake.append(img_embedding_fake.detach().cpu())
        img_real = real_imgs[-1]
        img_embedding_real = _get_img_embeddings(img_real, ImgEnc)
        img_feats_real.append(img_embedding_real.detach().cpu())

        if batch == 0:
            noise = torch.FloatTensor(args.batch_size, args.z_dim).normal_(0, 1).to(device)
            one_txt_feat = txt_embedding[0:1]
            one_txt_feat = one_txt_feat.repeat(args.batch_size, 1)
            fakes, _, _ = netG(noise, one_txt_feat)
            save_image(
                fakes[-1], 
                os.path.join(save_dir, 'random_noise_image0.jpg'), 
                normalize=True, scale_each=True)
    
    save_image(
            fake_imgs[0], 
            os.path.join(save_dir, 'batch{}_fake0.jpg'.format(batch)), 
            normalize=True, scale_each=True)
    save_image(
            fake_imgs[1], 
            os.path.join(save_dir, 'batch{}_fake1.jpg'.format(batch)), 
            normalize=True, scale_each=True)
    save_image(
            fake_imgs[2], 
            os.path.join(save_dir, 'batch{}_fake2.jpg'.format(batch)), 
            normalize=True, scale_each=True)
    save_image(
            real_imgs[-1], 
            os.path.join(save_dir, 'batch{}_real.jpg'.format(batch)), 
            normalize=True)

    real_fake = torch.stack([real_imgs[-1], fake_imgs[-1]]).permute(1,0,2,3,4).contiguous()
    real_fake = real_fake.view(-1, real_fake.shape[-3], real_fake.shape[-2], real_fake.shape[-1])
    save_image(
            real_fake, 
            os.path.join(save_dir, 'batch{}_real_fake.jpg'.format(batch)), 
            normalize=True, scale_each=True)
    batch += 1

txt_feats_real = torch.cat(txt_feats_real, dim=0)
img_feats_real = torch.cat(img_feats_real, dim=0)
img_feats_fake = torch.cat(img_feats_fake, dim=0)
cos = torch.nn.CosineSimilarity(dim=1)
dists = cos(txt_feats_real, img_feats_real)
print('=> Real txt and real img cosine (N={}): {:.4f}({:.4f})'.format(dists.shape[0], dists.mean().item(), dists.std().item()))
dists = cos(txt_feats_real, img_feats_fake)
print('=> Real txt and fake img cosine (N={}): {:.4f}({:.4f})'.format(dists.shape[0], dists.mean().item(), dists.std().item()))

N = min(1000, txt_feats_real.shape[0])

idxs = np.random.choice(img_feats_real.shape[0], N, replace=False)
sub = img_feats_real.numpy()[idxs]
Y = 1-pdist(sub, 'cosine')
print('=> Two random real images cosine (N={}): {:.4f}({:.4f})'.format(Y.shape[0], Y.mean().item(), Y.std().item()))

idxs = np.random.choice(txt_feats_real.shape[0], N, replace=False)
sub = txt_feats_real.numpy()[idxs]
Y = 1-pdist(sub, 'cosine')
print('=> Two random real texts cosine (N={}): {:.4f}({:.4f})'.format(Y.shape[0], Y.mean().item(), Y.std().item()))

idxs = np.random.choice(img_feats_fake.shape[0], N, replace=False)
sub = img_feats_fake.numpy()[idxs]
Y = 1-pdist(sub, 'cosine')
print('=> Two random fake images cosine (N={}): {:.4f}({:.4f})'.format(Y.shape[0], Y.mean().item(), Y.std().item()))


print('=> computing ranks...')
retrieved_range = min(900, len(dataloader)*args.batch_size)
medR, medR_std, recalls = rank(txt_feats_real.numpy(), img_feats_real.numpy(), retrieved_type='recipe', retrieved_range=retrieved_range)
print('=> Real MedR: {:.4f}({:.4f})'.format(medR, medR_std))
for k, v in recalls.items():
    print('Real Recall@{} = {:.4f}'.format(k, v))

medR, medR_std, recalls = rank(txt_feats_real.numpy(), img_feats_fake.numpy(), retrieved_type='recipe', retrieved_range=retrieved_range)
print('=> Fake MedR: {:.4f}({:.4f})'.format(medR, medR_std))
for k, v in recalls.items():
    print('Fake Recall@{} = {:.4f}'.format(k, v))