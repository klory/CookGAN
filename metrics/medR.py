import argparse
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import pdb
import os
import csv
from glob import glob
import math
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
import sys
sys.path.append('../retrieval_model')
from utils_retrieval import compute_statistics
import train_retrieval
sys.path.append('../')
from common import requires_grad

if __name__ == '__main__':
    from utils_metrics import load_args, normalize, resize
    args = load_args()

    # assertations
    assert 'ckpt_dir' in args.__dict__
    assert 'retrieval_model' in args.__dict__
    assert 'device' in args.__dict__
    assert 'batch_size' in args.__dict__

    sys.path.append('../cookgan/')
    from generate_batch import BatchGenerator    
    
    device = args.device
    _, _, txt_encoder, img_encoder, _ = train_retrieval.load_model(args.retrieval_model, device)
    requires_grad(txt_encoder, False)
    requires_grad(img_encoder, False)
    txt_encoder = txt_encoder.eval()
    img_encoder = img_encoder.eval()

    filename = os.path.join(args.ckpt_dir, 'medR.csv')

    # load values that are already computed
    computed = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                computed += [row[0]]
    
    # prepare to write
    f = open(filename, mode='a')
    writer = csv.writer(f, delimiter=',')

    # find checkpoints
    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt')) + glob(os.path.join(args.ckpt_dir, '*.pt'))+glob(os.path.join(args.ckpt_dir, '*.pth'))
    ckpt_paths = sorted(ckpt_paths)
    print('records:', ckpt_paths)
    print('computed:', computed)
    for ckpt_path in ckpt_paths:
        print()
        print(f'working on {ckpt_path}')
        iteration = os.path.basename(ckpt_path).split('.')[0]
        if iteration in computed:
            print('already computed')
            continue

        print('==> computing MedR')
        args.ckpt_path = ckpt_path
        batch_generator = BatchGenerator(args)

        txt_outputs = []
        img_outputs = []
        with torch.no_grad():
            for _ in tqdm(range(1000//args.batch_size+1)):
                # generate
                txt, fake_img = batch_generator.generate_MedR()
                # fake_img: normalize
                fake_img = normalize(fake_img)
                # fake_img: resize
                fake_img = resize(fake_img, size=224)
                # retrieve
                txt_output, _ = txt_encoder(*txt)
                img_output = img_encoder(fake_img)
                txt_outputs.append(txt_output.detach().cpu())
                img_outputs.append(img_output.detach().cpu())
        txt_outputs = torch.cat(txt_outputs, dim=0).numpy()
        img_outputs = torch.cat(img_outputs, dim=0).numpy()
        retrieved_range = min(txt_outputs.shape[0], 1000)
        medR, recalls = compute_statistics(
            txt_outputs, img_outputs, retrieved_type='image', 
            retrieved_range=retrieved_range, verbose=True)
        print(f'{iteration}, MedR={medR.mean()}')
        writer.writerow ([iteration, medR.mean()])

    f.close()
    medRs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            medR = float(row[1])
            medRs += [medR]
    fig = plt.figure(figsize=(6,6))
    plt.plot(medRs)
    plt.savefig(os.path.join(args.ckpt_dir, 'medR.png'))