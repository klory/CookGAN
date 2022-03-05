import argparse
import pickle
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception3
import numpy as np
from tqdm import tqdm

from inception import InceptionV3

import sys
sys.path.append('../')
import common
sys.path.append('../retrieval_model')
import train_retrieval
sys.path.append('../cookgan')
import train_cookgan
from utils_cookgan import compute_txt_feat
from datasets_cookgan import FoodDataset


class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for _, imgs, _, _ in pbar:
        img = imgs[-1].to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


if __name__ == '__main__':
    from utils_metrics import load_args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = load_args()

    _, _, txt_encoder, _, _ = train_retrieval.load_model(args.retrieval_model, device)
    txt_encoder = txt_encoder.eval().to(device)
    ckpt_args, _, netG, _, _, _ = train_cookgan.load_model(args.ckpt_path, device)
    netG = netG.eval().to(device)

    inception = load_patched_inception_v3()
    inception = nn.DataParallel(inception).eval().to(device)

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
        
    dataset_name = 'Recipe1M'
    if ckpt_args.food_type:
        dataset_name += f'_{ckpt_args.food_type}'
        
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    features = extract_features(loader, inception, device).numpy()

    features = features[: args.n_sample]

    print(f'extracted {features.shape[0]} features')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    with open(f'inception_{dataset_name}.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'dataset_name': dataset_name}, f)