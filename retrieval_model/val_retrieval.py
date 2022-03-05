import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import csv
from glob import glob

from args_retrieval import get_parser
from utils_retrieval import compute_statistics, compute_ranks
from datasets_retrieval import Dataset, val_transform
from train_retrieval import load_model

def extract_features(text_encoder, image_encoder, ckpt_args, data_loader):
    text_encoder.eval()
    image_encoder.eval()
    txt_feats = []
    img_feats = []
    if ckpt_args.text_info[0] == '1':
        title_attn = []
    if ckpt_args.text_info[1] == '1':
        ingr_attn = []
    if ckpt_args.text_info[2] == '1':
        inst_attn = []
        inst_word_attn = []
    for data in tqdm(data_loader):
        txt, img = data
        for i in range(len(txt)):
            txt[i] = txt[i].to(device)
        img = img.to(device)

        with torch.no_grad():
            txt_feat, attentions = text_encoder(*txt)
            if ckpt_args.with_attention:
                if ckpt_args.text_info[0] == '1':
                    title_attn.append(attentions[0])
                if ckpt_args.text_info[1] == '1':
                    ingr_attn.append(attentions[1])
                if ckpt_args.text_info[2] == '1':
                    inst_attn.append(attentions[2])
                    inst_word_attn.append(attentions[3])
                    
            img_feat = image_encoder(img)
            txt_feats.append(txt_feat.detach().cpu())
            img_feats.append(img_feat.detach().cpu())

    txt_feats = torch.cat(txt_feats, dim=0)
    img_feats = torch.cat(img_feats, dim=0)
    attentions = [None, None, None, None]
    if ckpt_args.with_attention:
        if ckpt_args.text_info[0] == '1':
            title_attn = torch.cat(title_attn, dim=0).cpu().numpy()
            attentions[0] = title_attn
        if ckpt_args.text_info[1] == '1':
            ingr_attn = torch.cat(ingr_attn, dim=0).cpu().numpy()
            attentions[1] = ingr_attn
        if ckpt_args.text_info[2] == '1':
            inst_attn = torch.cat(inst_attn, dim=0).cpu().numpy()
            attentions[2] = inst_attn

    return txt_feats.numpy(), img_feats.numpy(), attentions


if __name__ == '__main__':
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if not args.ckpt_dir:
        args.ckpt_dir = '/common/home/fh199/CookGAN/retrieval_model/wandb/run-20201202_174456-3kh60es7/files'

    filename = os.path.join(args.ckpt_dir, f'metrics.csv')
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
        
    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt'))
    ckpt_paths = sorted(ckpt_paths)
    print('records:', ckpt_paths)
    print('computed:', computed)
    data_loader = None
    w2i = None
    for ckpt_path in ckpt_paths:
        print()
        print(f'working on {ckpt_path}')

        ckpt_args, _, text_encoder, image_encoder, _ = load_model(ckpt_path, device)
        
        if not data_loader:
            print('loading dataset')
            dataset = Dataset(
                part='val', 
                recipe_file=ckpt_args.recipe_file,
                img_dir=ckpt_args.img_dir, 
                word2vec_file=ckpt_args.word2vec_file, 
                permute_ingrs=ckpt_args.permute_ingrs,
                transform=val_transform, 
            )
            w2i = dataset.w2i
            dataset = torch.utils.data.Subset(dataset, indices=np.random.choice(len(dataset), 5000))
            data_loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            print('data info:', len(dataset), len(data_loader))

        txt_feats, img_feats, attentions = extract_features(text_encoder, image_encoder, ckpt_args, data_loader)
        title_attn, ingr_attn, inst_attn, _ = attentions

        retrieved_range = min(txt_feats.shape[0], args.retrieved_range)
        medRs, recalls = compute_statistics(
            txt_feats, 
            img_feats, 
            retrieved_type=args.retrieved_type, 
            retrieved_range=retrieved_range,
            draw_hist=False)
        
        writer.writerow([ckpt_path, medRs.mean(), medRs.std(), recalls[1], recalls[5], recalls[10]])