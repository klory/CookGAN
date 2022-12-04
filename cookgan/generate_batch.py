import torch
from torch._C import device
from torchvision import transforms
from PIL import Image
import os
import lmdb
from torch.utils import data
import json
from io import BytesIO
import numpy as np

import sys
sys.path.append('/data/CS470_HnC/')
import common
sys.path.append('/data/CS470_HnC/retrieval_model/')
import train_retrieval
sys.path.append('/data/CS470_HnC/cookgan')
import train_cookgan
from utils_cookgan import compute_txt_feat
from datasets_cookgan import FoodDataset

class BatchGenerator():
    def __init__(self, args):
        device = args.device
        _, _, txt_encoder, _, _ = train_retrieval.load_model(args.retrieval_model, device)
        ckpt_args, _, netG, _, _, _ = train_cookgan.load_model(args.ckpt_path, device)
        netG = netG.eval().to(device)

        txt_encoder = txt_encoder.eval().to(device)
        
        imsize = ckpt_args.base_size * (2 ** (ckpt_args.levels-1))
        train_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.CenterCrop(imsize)])
        dataset = FoodDataset(
            recipe_file=ckpt_args.recipe_file,
            img_dir=ckpt_args.img_dir,
            levels=ckpt_args.levels,
            part='val', 
            food_type=ckpt_args.food_type,
            base_size=ckpt_args.base_size, 
            transform=train_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4)
        
        self.ckpt_args = ckpt_args
        self.netG = netG
        self.txt_encoder = txt_encoder
        self.dataloader = dataloader
        self.batch_size = args.batch_size
        self.device = device
        self.fixed_noise = torch.randn(self.batch_size, self.ckpt_args.z_dim).to(self.device)
    
    def generate_fid(self):
        _, _, batch_fake_img = self.generate_all()
        return batch_fake_img

    def generate_MedR(self):
        batch_txt, _, batch_fake_img = self.generate_all()
        return batch_txt, batch_fake_img

    def generate_all(self):
        batch_txt, batch_imgs, _, _ = next(common.sample_data(self.dataloader))
        with torch.no_grad():
            txt_feat = compute_txt_feat(batch_txt, self.txt_encoder)
            fakes, _, _ = self.netG(self.fixed_noise, txt_feat)
        return batch_txt, batch_imgs[-1], fakes[-1]
        # batch_txt: CookGAN txt input
        # batch_img: [BS, 3, size, size]
        # batch_fake_img: [BS, 3, size, size]
        

if __name__ == '__main__':
    import pdb
    from types import SimpleNamespace
    args = SimpleNamespace(
        ckpt_path='/data/CS470_HnC/cookgan/wandb/run-20221117_162130-2ezrqode/files/000000.ckpt',
        retrieval_model='/data/CS470_HnC/retrieval_model/wandb/run-20221115_141017-qn8zgvm8/files/00000000.ckpt',
        batch_size=16,
        size=256,
        device='cuda',
    )

    # Recipe1M
    batch_generator = BatchGenerator(args)
    
    txt, img, fake_img = batch_generator.generate_all()
    print(len(txt))
    for t in txt:
        print(t.shape)
    print(img.shape)
    print(fake_img.shape)
