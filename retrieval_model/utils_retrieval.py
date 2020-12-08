import os
import json
import numpy as np
import json
from matplotlib import pyplot as plt

def compute_statistics(rcps, imgs, retrieved_type='recipe', retrieved_range=1000, draw_hist=False, verbose=True):
    if verbose:
        print('retrieved_range =', retrieved_range)
    N = retrieved_range
    data_size = imgs.shape[0]
    glob_medR = []
    glob_recall = {1:0.0, 5:0.0, 10:0.0}
    if draw_hist:
        plt.figure(figsize=(16, 6))
    # average over 10 sets
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        imgs_sub = imgs_sub / np.linalg.norm(imgs_sub, axis=1)[:, None]
        rcps_sub = rcps_sub / np.linalg.norm(rcps_sub, axis=1)[:, None]
        if retrieved_type == 'recipe':
            queries = imgs_sub
            values = rcps_sub
        else:
            queries = rcps_sub
            values = imgs_sub
        ranks = compute_ranks(queries, values)
        recall = {1:0.0, 5:0.0, 10:0.0}
        recall[1] = (ranks<=1).sum() / N 
        recall[5] = (ranks<=5).sum() / N
        recall[10] = (ranks<=10).sum() / N
        medR = int(np.median(ranks))
        for ii in recall.keys():
            glob_recall[ii] += recall[ii]
        glob_medR.append(medR)
        if draw_hist:
            ranks = np.array(ranks)
            plt.subplot(2,5,i+1)
            n, bins, patches = plt.hist(x=ranks, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.ylim(top=300)
            # plt.xlabel('Rank')
            # plt.ylabel('Frequency')
            # plt.title('Rank Distribution')
            plt.text(23, 45, 'avgR(std) = {:.2f}({:.2f})\nmedR={:.2f}\n#<{:d}:{:d}|#={:d}:{:d}|#>{:d}:{:d}'.format(
                np.mean(ranks), np.std(ranks), np.median(ranks), 
                medR,(ranks<medR).sum(), medR,(ranks==medR).sum(), medR,(ranks>medR).sum()))
    if draw_hist:
        plt.savefig('hist.png')
    
    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    
    glob_medR = np.array(glob_medR)
    if verbose:
        print('MedR = {:.4f}({:.4f})'.format(glob_medR.mean(), glob_medR.std()))
        for k,v in glob_recall.items():
            print('Recall@{} = {:.4f}'.format(k, v))
    return glob_medR, glob_recall

def compute_ranks(queries, values):
    """compute the ranks for queries and values
    
    Arguments:
        queries {np.array} -- text feats (or image feats)
        values {np.array} -- image feats (or text feats)
    """
    sims = np.dot(queries, values.T)
    ranks = []
    # loop through the N similarities for images
    for ii in range(sims.shape[0]):
        # get a column of similarities for image ii
        sim = sims[ii,:]
        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()
        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)
        # store the position
        ranks.append(pos+1)
    return np.array(ranks)