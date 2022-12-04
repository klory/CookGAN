import json, pickle
import os
from glob import glob
import numpy as np
from torchvision import transforms
from torch.utils import data
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image

import sys
sys.path.append('/data/CS470_HnC/made_a_little_cookgan/')
from common_new import load_recipes, get_title_wordvec, get_ingredients_wordvec, get_instructions_wordvec

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print('error to open image:', path)
        return Image.new('RGB', (224, 224), 'white')

def choose_one_image(rcp, img_dir):    
    """
    Arguments:
        rcp: recipe
        img_dir: image directory
    Returns:
        PIL image
    """
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
    return default_loader(path)


class Dataset(data.Dataset):
    def __init__(
        self, 
        part, 
        recipe_file,
        img_dir, 
        word2vec_file, 
        transform=None, 
        permute_ingrs=False):
        assert part in ('train', 'val', 'test'), \
            'part must be one of [train, val, test]'
        self.recipes = load_recipes(recipe_file, part)
        
        wv = KeyedVectors.load(word2vec_file, mmap='r')
        w2i = {w: i+2 for i, w in enumerate(wv.index2word)}
        w2i['<other>'] = 1
        self.w2i = w2i
        print('vocab size =', len(self.w2i))

        self.img_dir = img_dir
        self.transform = transform
        self.permute_ingrs = permute_ingrs
    
    def _prepare_one_recipe(self, index):
        rcp = self.recipes[index]

        title, n_words_in_title = get_title_wordvec(rcp, self.w2i) # np.int [max_len]
        ingredients, n_ingrs = get_ingredients_wordvec(rcp, self.w2i, self.permute_ingrs) # np.int [max_len]
        instructions, n_insts, n_words_each_inst = get_instructions_wordvec(rcp, self.w2i) # np.int [max_len, max_len]

        pil_img = choose_one_image(rcp, self.img_dir) # PIL [3, 224, 224]
        if self.transform:
            img = self.transform(pil_img)
        return [title, n_words_in_title, ingredients, n_ingrs, instructions, n_insts, n_words_each_inst], img

    def __getitem__(self, index):
        txt, img = self._prepare_one_recipe(index)
        return txt, img
    
    def __len__(self):
        return len(self.recipes)


if __name__ == '__main__':
    dataset = Dataset(
        part='train', 
        recipe_file='../data/Recipe1M/recipes_withImage.json',
        img_dir='../data/Recipe1M/images', 
        word2vec_file='../retrieval_model/models/word2vec_recipes.bin', 
        transform=train_transform, 
        permute_ingrs=False)

    for data in dataset:
        txt, img = data
        i2w = {i:w for w,i in dataset.w2i.items()}
        def get_words(vec, length, i2w):
            words = []
            for i in vec[:length]:
                words.append(i2w[i])
            return words

        print('[title] = {}'.format(' '.join(get_words(txt[0], txt[1], i2w))))
        print('[ingredients] = {}'.format(get_words(txt[2], txt[3], i2w)))
        print('[instructions]')
        instructions, n_insts, n_words_each_inst = txt[4], txt[5], txt[6]
        for i in range(n_insts):
            inst = instructions[i]
            inst_length = n_words_each_inst[i]
            print('[{:>2d}] {}'.format(i+1, ' '.join(get_words(inst, inst_length, i2w))))

        from matplotlib import pyplot as plt
        def show(img):
            npimg = img.numpy()
            npimg = (npimg-npimg.min()) / (npimg.max()-npimg.min())
            plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
            plt.show()

        show(img)
        print()