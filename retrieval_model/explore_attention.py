# TODO: finish this script
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import csv
from glob import glob

from args_retrieval import get_parser
from utils_retrieval import compute_ranks
from datasets_retrieval import Dataset, val_transform
from train_retrieval import load_model
from val_retrieval import extract_features

if __name__ == '__main__':
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    assert args.ckpt_path!=''
    ckpt_dir = os.path.dirname(args.ckpt_path)

    ckpt_args, _, text_encoder, image_encoder, _ = load_model(args.ckpt_path, device)
    
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

    # draw attention if possible:
    def save_attention_result(index, dataset, i2w, ranks, ingr_attn, ckpt_dir):
        fig = plt.figure(figsize=(12,6))
        [title, n_words_in_title, ingredients, n_ingrs, instructions, n_insts, n_words_each_inst], img = dataset[index]
        ingr_alpha = ingr_attn[index]

        title_disp = ' '.join([i2w[idx] for idx in title[:n_words_in_title]])
        fig.suptitle(title_disp)

    #     # title
    #     one_vector = title[i]
    #     one_alpha = alpha_title[i]
    #     length = len(one_vector.nonzero()[0])
    #     one_word_list = [i2w[idx] for idx in one_vector[:length]]
    #     ind = np.arange(length)
    #     plt.subplot(411)
    #     # pdb.set_trace()
    #     plt.barh(ind, one_alpha[:length])
    #     plt.yticks(ind, one_word_list)

        # ingredients
        one_vector = ingredients
        one_alpha = ingr_alpha
        one_word_list = [i2w[idx] for idx in one_vector[:n_ingrs]]
        ind = np.arange(n_ingrs)
        plt.subplot(121)
        plt.barh(ind, one_alpha[:n_ingrs])
        plt.yticks(ind, one_word_list, fontsize=12)

        # # instructions
        # one_matrix = instructions[i]
        # one_alpha = alpha_instructions[i]
        # # pdb.set_trace()
        # length = one_matrix.nonzero()[0].max() + 1
        # one_sentence_list = []
        # for k in range(length):
        #     one_vector = one_matrix[k]
        #     one_vector_length = len(one_vector.nonzero()[0])
        #     one_sentence = ' '.join([i2w[idx] for idx in one_vector[:one_vector_length]])
        #     one_sentence_list.append(one_sentence)
        # ind = np.arange(length)
        # plt.subplot(413)
        # plt.barh(ind, one_alpha[:length])
        # plt.yticks(ind, one_sentence_list)

        # images
        plt.subplot(122)
        one_img = img.permute(1,2,0).detach().cpu().numpy()
        scale = one_img.max() - one_img.min()
        one_img = (one_img - one_img.min()) / scale
        plt.imshow(one_img)
        plt.axis('off')
        
        plt.savefig(os.path.join(ckpt_dir, 'rank={}_{}.jpg'.format(ranks[index], title_disp)))

    if ckpt_args.with_attention:
        from matplotlib import pyplot as plt
        ranks = compute_ranks(txt_feats[:1000], img_feats[:1000])

        print('plot ranks')
        medR = np.median(ranks).astype(int)
        plt.figure(figsize=(6,6))
        n, bins, patches = plt.hist(x=ranks, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title('Rank Distribution')
        plt.text(23, 45, 'avgR(std) = {:.2f}({:.2f})\nmedR={:d}\n#<{:d}:{:d}|#={:d}:{:d}|#>{:d}:{:d}'.format(
            np.mean(ranks), np.std(ranks), medR, 
            medR,(ranks<medR).sum(), medR,(ranks==medR).sum(), medR,(ranks>medR).sum()))
        plt.savefig(os.path.join(args.ckpt_dir, 'batch_ranks.jpg'))

        print('plot attentions')
        ingr_attn = ingr_attn[:1000]
        sorted_idxs = np.argsort(ranks).tolist()
        i2w = {i:w for w,i in w2i.items()}
        for i in range(10):
            idx = sorted_idxs[i]
            save_attention_result(idx, dataset, i2w, ranks, ingr_attn, args.ckpt_dir)
        for i in range(-10, 0):
            idx = sorted_idxs[i]
            save_attention_result(idx, dataset, i2w, ranks, ingr_attn, args.ckpt_dir)