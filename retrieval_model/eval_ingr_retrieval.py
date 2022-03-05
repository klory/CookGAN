# TODO: finish this script
'''
CUDA_VISIBLE_DEVICES=0 python eval_ingr_retrieval.py \
    --batch_size=32 --resume=models/010.ckpt \
    --food_type=salad --hot_ingr=tomato --save_dir=experiments \
    --generation_model=generative_model/models/salad.ckpt
'''

import torch
from torch import nn
from torch.nn import functional as F
from networks import TextEncoder, ImageEncoder
from dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from args import args
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
from utils import load_recipes, load_dict, load_retrieval_model, load_generation_model
from utils import compute_img_feature, compute_txt_feature, transform
from inflection import singularize, pluralize

# type_ = 'salad'
# hot_ingr = ['tomato', 'cucumber', 'black_olife', 'avocado', 'carrot', 'red_pepper']

# type_ = 'cookie'
# hot_ingr = ['walnut', 'chocolate', 'coconut', 'molass', 'orange']

# type_ = 'muffin'
# hot_ingr = ['blueberry', 'chocolate', 'oat', 'banana', 'cranberry']

assert args.resume != ''

food_type = args.food_type
hot_ingr = args.hot_ingr
tops = 5

generation_model_name = args.generation_model.rsplit('/', 1)[-1].rsplit('.', 1)[0]
save_dir = os.path.join(args.save_dir, generation_model_name)
if not os.path.exists(save_dir):
    print('create directory:', save_dir)
    os.makedirs(save_dir)

recipes = load_recipes(os.path.join(args.data_dir,'recipesV1.json'), 'val')
recipes = [x for x in recipes if food_type.lower() in x['title'].lower()]
print('# {} recipes:'.format(food_type), len(recipes))
vocab_inst = load_dict(os.path.join(args.data_dir, 'vocab_inst.txt'))
print('#vocab_inst:', len(vocab_inst))
vocab_ingr = load_dict(os.path.join(args.data_dir, 'vocab_ingr.txt'))
print('#vocab_ingr:', len(vocab_ingr))

for recipe in recipes:
    ingrediens_list = recipe['ingredients']
    recipe['new_ingrs'] = []
    for name in ingrediens_list:
        if hot_ingr in name:
            recipe['new_ingrs'].append(hot_ingr)
        else:
            recipe['new_ingrs'].append(name)

device = torch.device('cuda' \
    if torch.cuda.is_available() and args.cuda
    else 'cpu')
print('device:', device)
if device.__str__() == 'cpu':
    args.batch_size = 16

TxtEnc, ImgEnc = load_retrieval_model(args.resume, device)

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
ids_b = set()
uniques_b = []
for rcp in recipes_b:
    if rcp['id'] not in ids_b:
        uniques_b.append(rcp)
        ids_b.add(rcp['id'])
print('#unique = {}, #unique_ = {}'.format(len(uniques_a), len(uniques_b)))

#####################################
print('-' * 40)
print('compute REAL image features for interesting recipes')
#####################################
if len(uniques_a)>1 and len(uniques_b)>1:
    print('compute REAL image features for recipes with {}'.format(hot_ingr))
    img_a, img_feat_a = compute_img_feature(uniques_a, args.img_dir, ImgEnc, transform, device)
    save_image(
        img_a[:64], 
        os.path.join(save_dir, '{}_with.jpg'.format(hot_ingr)), 
        normalize=True)
    img_feat_a = img_feat_a.detach().cpu().numpy()
    
    print('compute REAL image features for recipes without {}'.format(hot_ingr))
    img_b, img_feat_b = compute_img_feature(uniques_b, args.img_dir, ImgEnc, transform, device)
    save_image(
        img_b[:64], 
        os.path.join(save_dir, '{}_without.jpg'.format(hot_ingr)),
        normalize=True)
    img_feat_b = img_feat_b.detach().cpu().numpy()
else:
    print('unable to compute')
    sys.exit(-1)


# ******************************************************
print('-' * 40)
print('compute text features for all recipes')
# *****************************************************
_, txt_feats = compute_txt_feature(recipes, TxtEnc, vocab_inst, vocab_ingr, device)
txt_feats = txt_feats.cpu().numpy()

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
netG = load_generation_model(args.generation_model, device)

with open(os.path.join(save_dir, '{}_with.json'.format(hot_ingr)), 'w') as f:
    json.dump(uniques_a, f, indent=2)
with open(os.path.join(save_dir, '{}_without.json'.format(hot_ingr)), 'w') as f:
    json.dump(uniques_b, f, indent=2)
N = min(len(uniques_a), len(uniques_b))
N = min(N, 128)
print('compute text features for recipes with {}'.format(hot_ingr))
_, txt_feat_y = compute_txt_feature(uniques_a[:N], TxtEnc, vocab_inst, vocab_ingr, device)
print('compute text features for recipes without {}'.format(hot_ingr))
_, txt_feat_n = compute_txt_feature(uniques_b[:N], TxtEnc, vocab_inst, vocab_ingr, device)
interpolate_points = [1.0, 0.75, 0.5, 0.25, 0.0]
print('interpolate points:', interpolate_points)
if args.cuda:
    fixed_noise = torch.zeros(1, 100).to(device)
    fixed_noise = fixed_noise.repeat(N, 1)

imgs_all = []
for w_y in interpolate_points:
    txt_embedding = w_y*txt_feat_y + (1-w_y)*txt_feat_n
    with torch.no_grad():
        fake_imgs, _, _ = netG(fixed_noise, txt_embedding)
    imgs = fake_imgs[-1] # those 256x256 images
    imgs_all.append(imgs[:12])
    imgs = imgs/2 + 0.5
    imgs = F.interpolate(imgs, [224, 224], mode='bilinear', align_corners=True)
    for i in range(imgs.shape[1]):
        imgs[:,i] = (imgs[:,i]-mean[i])/std[i]
    with torch.no_grad():
        img_feats = ImgEnc(imgs).detach().cpu().numpy()
    cvgs = compute_ingredient_retrival_score(img_feats, txt_feats, tops)
    print('with/without={:.2f}/{:.2f}, avg cvg (over {} recipes)={:.2f} ({:.2f})'.format(
        w_y, 1-w_y, N, cvgs.mean(), cvgs.std()))

imgs_all = torch.stack(imgs_all) # [5, 8, 3, 256, 256]
imgs_all = imgs_all.permute(1,0,2,3,4).contiguous() # [8, 5, 3, 256, 256]
imgs_all = imgs_all.view(-1, 3, 256, 256) # [40, 3, 256, 256]
save_image(
    imgs_all, 
    os.path.join(save_dir, '{}_interpolations.jpg'.format(hot_ingr)), 
    nrow=5, 
    normalize=True, 
    scale_each=True)