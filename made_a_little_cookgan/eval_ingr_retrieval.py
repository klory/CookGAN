'''
CUDA_VISIBLE_DEVICES=0 python eval_ingr_retrieval.py --batch_size=32
'''

import torch
from torch import nn
from torch.nn import functional as F
from datasets_retrieval import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import args_retrieval
from tqdm import tqdm
import numpy as np
import os
import sys
import math
import pdb
from copy import deepcopy
from glob import glob
from PIL import Image
import json
from common_new import load_recipes
from common_new import load_dict
from common_new import compute_img_feature, compute_txt_feature

#추가
import models_retrieval_nobak
import models_cookgan_for_retrieval

# type_ = 'salad'
# hot_ingr = ['tomato', 'cucumber', 'black_olive', 'carrot', 'red_pepper']

parser = args_retrieval.get_parser()
args = parser.parse_args()

food_type = 'salad'
hot_ingr = 'carrots'
tops = 5

data_dir = '/data/CS470_HnC/made_a_little_cookgan/interpolation_example/data'
save_dir = '/data/CS470_HnC/made_a_little_cookgan/interpolation_example/save'
img_dir = '/data/CS470_HnC/made_a_little_cookgan/interpolation_example/img'
if not os.path.exists(save_dir):
    print('create directory:', save_dir)
    os.makedirs(save_dir)

print('loading recipes and vocab...')

recipes = load_recipes(os.path.join(data_dir,'recipes.json'), 'val')
recipes = [x for x in recipes if food_type.lower() in x['title'].lower()]
print('# {} recipes:'.format(food_type), len(recipes))
vocab_inst = load_dict(os.path.join(data_dir, 'vocab_inst.txt'))
print('#vocab_inst:', len(vocab_inst))
vocab_ingr = load_dict(os.path.join(data_dir, 'vocab_ingr.txt'))
print('#vocab_ingr:', len(vocab_ingr))

for recipe in recipes:
    ingrediens_list = recipe['ingredients']
    recipe['new_ingrs'] = []
    for name in ingrediens_list:
        if hot_ingr in name:
            recipe['new_ingrs'].append(hot_ingr)
        else:
            recipe['new_ingrs'].append(name)

### Txtenc 받는 부분 수정
TxtEnc = models_retrieval_nobak.TextEncoder(
    data_dir='/data/CS470_HnC/made_a_little_cookgan/', text_info='010', hid_dim=300,
    emb_dim=300, z_dim=1024, with_attention=2,
    ingr_enc_type='rnn').eval()
TxtEnc.load_state_dict(torch.load('/data/CS470_HnC/made_a_little_cookgan/text_encoder.model'))
###


### Imgenc 받는 부분 수정
ImgEnc = models_retrieval_nobak.ImageEncoder(
        z_dim=1024).eval()
model = torch.load('/data/CS470_HnC/retrieval_model/wandb/run-20221115_141017-qn8zgvm8/files/00390000.ckpt')['image_encoder']
ImgEnc.load_state_dict(model, strict = False)
###

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

print('\nhot_ingr:', hot_ingr)
hot_recipes = [x for x in recipes if hot_ingr in x['new_ingrs']]
cold_recipes = [x for x in recipes if hot_ingr not in x['new_ingrs']]
print('#with={}/{} = {:.2f}, #without={}/{} = {:.2f}'.format(
    len(hot_recipes), len(recipes), 1.0*len(hot_recipes)/len(recipes), 
    len(cold_recipes), len(recipes), 1.0*len(cold_recipes)/len(recipes)))

recipes_a = []
recipes_b = []
threshold = 0.7
for rcp_a in hot_recipes:
    tmp = deepcopy(rcp_a['new_ingrs'])
    tmp.remove(hot_ingr)
    ingrs_a = set(tmp)
    for rcp_b in cold_recipes:
        ingrs_b = set(rcp_b['new_ingrs'])
        union = ingrs_a.union(ingrs_b)
        common = ingrs_a.intersection(ingrs_b)
        if 1.0*len(common)/len(union) >= threshold:
            recipes_a.append(rcp_a)
            recipes_b.append(rcp_b)
print('#{} pairs (IoU={:.2f}) = {}'.format(hot_ingr, threshold, len(recipes_a)))

ids_a = set()
uniques_a = []
for rcp in recipes_a:
    if rcp['id'] not in ids_a:
        uniques_a.append(rcp)
        ids_a.add(rcp['id'])
        print('a:', rcp['new_ingrs'])
ids_b = set()
uniques_b = []
for rcp in recipes_b:
    if rcp['id'] not in ids_b:
        uniques_b.append(rcp)
        ids_b.add(rcp['id'])
        print('b:', rcp['new_ingrs'])

print('#unique = {}, #unique_ = {}'.format(len(uniques_a), len(uniques_b)))

#####################################
print('-' * 40)
print('compute REAL image features for interesting recipes')
#####################################
if len(uniques_a)>1 and len(uniques_b)>1:
    print('compute REAL image features for recipes with {}'.format(hot_ingr))
    img_a, img_feat_a = compute_img_feature(uniques_a, ImgEnc)
    save_image(
        img_a[:64], 
        os.path.join(save_dir, '{}_with.jpg'.format(hot_ingr)), 
        normalize=True)
    img_feat_a = img_feat_a.detach().cpu().numpy()
    
    print('compute REAL image features for recipes without {}'.format(hot_ingr))
    img_b, img_feat_b = compute_img_feature(uniques_b, ImgEnc)
    save_image(
        img_b[:64], 
        os.path.join(save_dir, '{}_without.jpg'.format(hot_ingr)),
        normalize=True)
    img_feat_b = img_feat_b.detach().cpu().numpy()
else:
    print('unable to compute')
    sys.exit(-1)

print(uniques_a[:35])
print(uniques_b[:35])


# ******************************************************
print('-' * 40)
print('compute text features for all recipes')
# *****************************************************
print(recipes[0])
txt_feats = compute_txt_feature(recipes, TxtEnc, vocab_inst, vocab_ingr)
txt_feats = txt_feats.detach().cpu().numpy()
# txt_feats = txt_feats.cpu().numpy()

# ******************************************************
print('-' * 40)
print('compute coverage among the top {} retrieved recipes'.format(tops))
# *****************************************************
def compute_ingredient_retrival_score(imgs, txts, tops):
    imgs = imgs / np.linalg.norm(imgs, axis=1)[:, None]
    txts = txts / np.linalg.norm(txts, axis=1)[:, None]
    # retrieve recipe
    sims = np.dot(imgs, txts.T) # [N, N]
    # loop through the N similarities for images
    cvgs = []
    for ii in range(imgs.shape[0]):
        # get a row of similarities for image ii
        sim = sims[ii,:]
        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()
        topk_idxs = sorting[:tops]
        print(topk_idxs)
        success = 0.0
        for rcp_idx in topk_idxs:
            rcp = recipes[rcp_idx]
            ingrs = rcp['new_ingrs']
            if hot_ingr in ingrs:
                success += 1
        cvgs.append(success / tops)
    return np.array(cvgs)

cvgs = compute_ingredient_retrival_score(img_feat_a, txt_feats, tops)
print('Top {} avg coverage with {} (#={}) = {:.2f} ({:.2f})'.format(
    tops, hot_ingr, len(uniques_a), cvgs.mean(), cvgs.std()))
cvgs = compute_ingredient_retrival_score(img_feat_b, txt_feats, tops)
print('Top {} avg coverage without {} (#={}) = {:.2f} ({:.2f})'.format(
    tops, hot_ingr, len(uniques_b), cvgs.mean(), cvgs.std()))


# ******************************************************
print('-' * 40)
print('compute coverage for interpolation between with and without {}'.format(hot_ingr))
# *****************************************************
print('load pretrained generative model')

### netG 받는 부분 수정
netG = models_cookgan_for_retrieval.G_NET(levels=3).eval().requires_grad_(False)
netG.load_state_dict(torch.load('/data/CS470_HnC/made_a_little_cookgan/gen_salad_cycleTxt1.0_e300.model'))
###


with open(os.path.join(save_dir, '{}_with.json'.format(hot_ingr)), 'w') as f:
    json.dump(uniques_a, f, indent=2)
with open(os.path.join(save_dir, '{}_without.json'.format(hot_ingr)), 'w') as f:
    json.dump(uniques_b, f, indent=2)
N = min(len(uniques_a), len(uniques_b))
N = min(N, 128)
print('compute text features for recipes with {}'.format(hot_ingr))
txt_feat_y = compute_txt_feature(uniques_a[:N], TxtEnc, vocab_inst, vocab_ingr)
print('compute text features for recipes without {}'.format(hot_ingr))
txt_feat_n = compute_txt_feature(uniques_b[:N], TxtEnc, vocab_inst, vocab_ingr)
interpolate_points = [1.25, 1.0, 0.75, 0.5, 0.25, 0.0,-0.25] #test interpolation points
print('interpolate points:', interpolate_points)
fixed_noise = torch.zeros(1, 100)
fixed_noise = fixed_noise.repeat(N, 1)

imgs_all = []
for w_y in interpolate_points:
    txt_embedding = w_y*txt_feat_y + (1-w_y)*txt_feat_n
    with torch.no_grad():
        fake_imgs, _, _ = netG(fixed_noise, txt_embedding)
    imgs = fake_imgs[-1] # those 256x256 images
    print(len(imgs))
    imgs_all.append(imgs[:35])
    imgs = imgs/2 + 0.5
    imgs = F.interpolate(imgs, [224, 224], mode='bilinear', align_corners=True)
    for i in range(imgs.shape[1]):
        imgs[:,i] = (imgs[:,i]-mean[i])/std[i]
    with torch.no_grad():
        img_feats = ImgEnc(imgs).detach().cpu().numpy()
    cvgs = compute_ingredient_retrival_score(img_feats, txt_feats, tops)
    print('with/without={:.2f}/{:.2f}, avg cvg (over {} recipes)={:.2f} ({:.2f})'.format(
        w_y, 1-w_y, N, cvgs.mean(), cvgs.std()))

imgs_all = torch.stack(imgs_all) 
imgs_all = imgs_all.permute(1,0,2,3,4).contiguous() 
imgs_all = imgs_all.view(-1, 3, 256, 256)
save_image(
    imgs_all, 
    os.path.join(save_dir, '{}_interpolations.jpg'.format(hot_ingr)), 
    nrow=7, 
    normalize=True, 
    scale_each=True)