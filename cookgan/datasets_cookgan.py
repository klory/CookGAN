import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
import pickle
import numpy as np
import os
import json
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image

import sys
sys.path.append('../')
from common import load_recipes, get_title_wordvec, get_ingredients_wordvec_withClasses, get_instructions_wordvec

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None, levels=3):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(levels):
        if i < (levels - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

def choose_one_image_path(rcp, img_dir):
    part = rcp['partition']
    image_infos = rcp['images']
    if part == 'train':
        # We do only use the first five images per recipe during training
        imgIdx = np.random.choice(range(min(5, len(image_infos))))
    else:
        imgIdx = 0

    loader_path = [image_infos[imgIdx]['id'][i] for i in range(4)]
    loader_path = os.path.join(*loader_path)
    if 'plus' in img_dir:
        path = os.path.join(img_dir, loader_path, image_infos[imgIdx]['id'])
    else:
        path = os.path.join(img_dir, part, loader_path, image_infos[imgIdx]['id'])
    return path


class FoodDataset(data.Dataset):
    def __init__(
        self, 
        recipe_file, 
        img_dir,
        levels=3, 
        word2vec_file='../retrieval_model/models/word2vec_recipes.bin',
        vocab_ingrs_file='../manual_files/list_of_merged_ingredients.txt',
        part='train', 
        food_type='salad',
        base_size=64, 
        transform=None,
        num_samples=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        self.levels = levels
        self.recipe_file = recipe_file
        self.img_dir = img_dir
        for _ in range(levels):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.recipes = load_recipes(recipe_file, part)
        if food_type:
            self.recipes = [x for x in self.recipes if food_type.lower() in x['title'].lower()]
        if num_samples:
            N = min(len(self.recipes), num_samples)
            self.recipes = np.random.choice(self.recipes, N, replace=False)

        wv = KeyedVectors.load(word2vec_file, mmap='r')
        w2i = {w: i+2 for i, w in enumerate(wv.index2word)}
        w2i['<other>'] = 1
        self.w2i = w2i

        with open(vocab_ingrs_file, 'r') as f:
            vocab_ingrs = f.read().strip().split('\n')
            self.ingr2i = {ingr:i for i,ingr in enumerate(vocab_ingrs)}

    def __getitem__(self, index):
        rcp = self.recipes[index]

        title, n_words_in_title = get_title_wordvec(rcp, self.w2i) # np.int [max_len]
        ingredients, n_ingrs, _ = get_ingredients_wordvec_withClasses(rcp, self.w2i, self.ingr2i) # np.int [max_len]
        instructions, n_insts, n_words_each_inst = get_instructions_wordvec(rcp, self.w2i) # np.int [max_len, max_len]
        txt = (title, n_words_in_title, ingredients, n_ingrs, instructions, n_insts, n_words_each_inst)

        img_name = choose_one_image_path(rcp, self.img_dir)
        imgs = get_imgs(img_name, self.imsize, transform=self.transform, normalize=self.norm, levels=self.levels)

        all_idx = range(len(self.recipes))
        wrong_idx = np.random.choice(all_idx)
        while wrong_idx == index:
            wrong_idx = np.random.choice(all_idx)
        wrong_img_name = choose_one_image_path(self.recipes[wrong_idx], self.img_dir)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize, transform=self.transform, normalize=self.norm, levels=self.levels)

        return txt, imgs, wrong_imgs, rcp['title']

    def __len__(self):
        return len(self.recipes)

if __name__ == '__main__':
    class Args: pass
    args = Args()
    args.base_size = 64
    args.levels = 3
    args.recipe_file = '../data/Recipe1M/recipes_withImage.json'
    args.img_dir = '../data/Recipe1M/images'
    args.food_type = 'salad'
    args.batch_size = 32
    args.workers = 4

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
        drop_last=False, shuffle=False, num_workers=int(args.workers))

    for txt, imgs, w_imgs, title in train_loader:
        print(len(txt))
        for one_txt in txt:
            print(one_txt.shape)
        
        print(len(imgs))
        for img in imgs:
            print(img.shape)
        
        print(len(w_imgs))
        for img in w_imgs:
            print(img.shape)

        print(len(title))
        print(title[0])
        input()