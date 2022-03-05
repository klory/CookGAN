import os
import string
import json
import numpy as np
import re
import copy
from datetime import datetime
import json
import argparse

root = '/common/home/fh199/CookGAN'

def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def sample_data(loader):
    """
    arguments:
        loader: torch.utils.data.DataLoader
    return:
        one batch of data
    usage:
        data = next(sample_data(loader))
    """
    while True:
        for batch in loader:
            yield batch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT,ext)

class Layer(object):
    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json',ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base

def remove_numbers(s):
    '''
    remove numbers in a sentence.
    - 1.1:  \d+\.\d+
    - 1 1/2 or 1-1/2 or 1 -1/2 or 1- 1/2 or 1 - 1/2: (\d+ *-* *)?\d+/\d+
    - 1: \d+'
    
    Arguments:
        s {str} -- the string to operate on
    
    Returns:
        str -- the modified string without numbers
    '''
    return re.sub(r'\d+\.\d+|(\d+ *-* *)?\d+/\d+|\d+', 'some', s)

def tok(text, ts=False):
    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']
    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text


param_counter = lambda params: sum(p.numel() for p in params if p.requires_grad)


def load_recipes(file_path, part=None):
    with open(file_path, 'r') as f:
        info = json.load(f)
    if part:
        info = [x for x in info if x['partition']==part]
    return info


def get_title_wordvec(recipe, w2i, max_len=20):
    '''
    get the title wordvec for the recipe, the 
    number of items might be different for different 
    recipe
    '''
    title = recipe['title']
    words = title.split()
    vec = np.zeros([max_len], dtype=np.int)
    num_words = min(max_len, len(words))
    for i in range(num_words):
        word = words[i]
        if word not in w2i:
            word = '<other>'
        vec[i] = w2i[word]
    return vec, num_words


def get_instructions_wordvec(recipe, w2i, max_len=20):
    '''
    get the instructions wordvec for the recipe, the 
    number of items might be different for different 
    recipe
    '''
    instructions = recipe['instructions']
    # each recipe has at most max_len sentences
    # each sentence has at most max_len words
    vec = np.zeros([max_len, max_len], dtype=np.int)
    num_insts = min(max_len, len(instructions))
    num_words_each_inst = np.zeros(max_len, dtype=np.int)
    for row in range(num_insts):
        inst = instructions[row]
        words = inst.split()
        num_words = min(max_len, len(words))
        num_words_each_inst[row] = num_words
        for col in range(num_words):
            word = words[col]
            if word not in w2i:
                word = '<other>'
            vec[row, col] = w2i[word]
    return vec, num_insts, num_words_each_inst


def get_ingredients_wordvec(recipe, w2i, permute_ingrs=False, max_len=20):
    '''
    get the ingredients wordvec for the recipe, the 
    number of items might be different for different 
    recipe
    '''
    ingredients = recipe['ingredients']
    if permute_ingrs:
        ingredients = np.random.permutation(ingredients).tolist()
    vec = np.zeros([max_len], dtype=np.int)
    num_words = min(max_len, len(ingredients))
        
    for i in range(num_words):
        word = ingredients[i]
        if word not in w2i:
            word = '<other>'
        vec[i] = w2i[word]
        
    return vec, num_words


def get_ingredients_wordvec_withClasses(recipe, w2i, ingr2i, permute_ingrs=False, max_len=20):
    '''
    get the ingredients wordvec for the recipe, the 
    number of items might be different for different 
    recipe
    '''
    ingredients = recipe['ingredients']
    if permute_ingrs:
        ingredients = np.random.permutation(ingredients).tolist()

    label = np.zeros([len(ingr2i)], dtype=np.float32)

    vec = np.zeros([max_len], dtype=np.int)
    num_words = min(max_len, len(ingredients))
        
    for i in range(num_words):
        word = ingredients[i]
        if word not in w2i:
            word = '<other>'
        vec[i] = w2i[word]

        if word in ingr2i:
            label[ingr2i[word]] = 1
        
    return vec, num_words, label

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag