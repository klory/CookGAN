import yaml
from types import SimpleNamespace
import argparse
from torch.nn import functional as F
import sys
sys.path.append('../')
import common

def load_args():
    parser = argparse.ArgumentParser(description='Calculate Inception v3 features for datasets')
    parser.add_argument('--config', type=str, default=f'{common.root}/metrics/configs/salad+cookgan.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        args = SimpleNamespace(**data)
    return args


def normalize(img):
    img = (img-img.min())/(img.max()-img.min())
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:,i] = (img[:,i]-means[i])/stds[i]
    return img

def resize(img, size=224):
    return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
