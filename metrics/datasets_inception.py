from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import math
import torch
import json
import numpy as np
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])


class Recipe1MDataset(Dataset):
    def __init__(
        self, 
        lmdb_file='/dresden/users/fh199/food_project/data/Recipe1M/Recipe1M.lmdb',
        food_type='', transform=transform, resolution=256):

        assert food_type in ['', 'salad', 'cookie', 'muffin'], "part has to be in ['', 'salad', 'cookie', 'muffin']"

        dirname = os.path.dirname(lmdb_file)
        path = os.path.join(dirname, 'keys.json')
        with open(path, 'r') as f:
            self.keys = json.load(f)
        if food_type:
            self.keys = [x for x in self.keys if food_type.lower() in x['title'].lower()]

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        self.resolution = resolution

        assert transform!=None, 'transform can not be None!'
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def _load_recipe(self, rcp):
        rcp_id = rcp['id']

        with self.env.begin(write=False) as txn:
            key = f'title-{rcp_id}'.encode('utf-8')
            title = txn.get(key).decode('utf-8')

            key = f'ingredients-{rcp_id}'.encode('utf-8')
            ingredients = txn.get(key).decode('utf-8')

            key = f'instructions-{rcp_id}'.encode('utf-8')
            instructions = txn.get(key).decode('utf-8')

            key = f'{self.resolution}-{rcp_id}'.encode('utf-8')
            img_bytes = txn.get(key)

        txt = title
        txt += '\n'
        txt += ingredients
        txt += '\n'
        txt += instructions

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        return txt, img

    def __getitem__(self, index):
        rcp = self.keys[index]
        txt, img = self._load_recipe(rcp)

        return txt, img


class PizzaGANDataset(Dataset):
    def __init__(
        self, 
        lmdb_file='/dresden/users/fh199/food_project/data/pizzaGANdata_new_concise/pizzaGANdata.lmdb', 
        transform=transform, resolution=64):

        dirname = os.path.dirname(lmdb_file)
        label_file = os.path.join(dirname, 'imageLabels.txt')
        with open(label_file, 'r') as f:
            self.labels = f.read().strip().split('\n')
        
        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution

        assert transform!=None, 'transform can not be None!'
        self.transform = transform

    def __len__(self):
        return self.length

    def _load_pizza(self, idx):
        with self.env.begin(write=False) as txn:
            key = f'{idx}'.encode('utf-8')
            ingrs = txn.get(key).decode('utf-8')
            if not ingrs:
                ingrs = 'empty'
            key = f'{self.resolution}-{idx}'.encode('utf-8')
            img_bytes = txn.get(key)
        
        txt = ingrs
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        return txt, img

    def __getitem__(self, idx):
        txt, img = self._load_pizza(idx)
        return txt, img


if __name__ == '__main__':
    import pdb
    from matplotlib import pyplot as plt

    def show(img):
        npimg = img.numpy()
        npimg = (npimg-npimg.min()) / (npimg.max()-npimg.min())
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()
    
    res = 256

    dataset = PizzaGANDataset(
        lmdb_file='/dresden/users/fh199/food_project/data/pizzaGANdata_new_concise/pizzaGANdata.lmdb', 
        transform=transform, resolution=res)
    
    # dataset = Recipe1MDataset(
    #     lmdb_file='/dresden/users/fh199/food_project/data/Recipe1M/Recipe1M.lmdb',
    #     food_type='', transform=transform, resolution=res)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False)
    print(len(dataset), len(dataloader))
    for txt, img in dataloader:
        print(len(txt))
        print(txt[0])
        print(img.shape)
        # show(img)
        pdb.set_trace()
        break