import os
from tqdm import tqdm
import json
import argparse
import pickle
import re
from common import root, tok, remove_numbers
import common
import numpy as np

parser = argparse.ArgumentParser(
    description='clean recipes')
parser.add_argument(
    '--data_dir', default=f'{root}/data/Recipe1M', 
    help='the folder which contains Recipe1M text files')
parser.add_argument("--lower", type=int,  default=0, choices=[0,1])
parser.add_argument("--remove_numbers", type=int, default=0, choices=[0,1])
args = parser.parse_args()
data_dir = args.data_dir

print('load recipes (20 seconds)')
recipes_original = common.Layer.merge(
    [common.Layer.L1, common.Layer.L2, common.Layer.INGRS], 
    os.path.join(data_dir, 'texts'))

for rcp in recipes_original:
    rcp['instructions'] = [x['text'] for x in rcp['instructions']]
    rcp['ingredients'] = [x['text'] for x in rcp['ingredients']]

with open(f'{root}/manual_files/replacement_dict.pkl', 'rb') as f:
    replace_dict = pickle.load(f)

print('start processing')
cvgs = []
recipes = []
recipes_withImage = []
for rcp in tqdm(recipes_original):
    insts = []
    for inst in rcp['instructions']:
        # words = tok(inst['text']).split()
        words = tok(inst).split()
        inst_ = ' '.join(words)
        insts.append(inst_)
    insts = '\n'.join(insts)
    if len(insts) == 0:
        continue
    
    title = rcp['title']
    words = tok(title).split()
    title = ' '.join(words)

    if args.lower:
        insts = insts.lower()
        title = title.lower()
    if args.remove_numbers:
        insts = remove_numbers(insts)
        title = remove_numbers(title)
    
    ingrs = []
    N = len(rcp['ingredients'])
    n = 0
    for ingr in rcp['ingredients']:
        # 1. add 'space' before and after 12 punctuation
        # 2. change 'space' to 'underscore'
        # ingr_name = ingr['text']
        ingr_name = ingr
        name = re.sub(' +', ' ', tok(ingr_name)).replace(' ', '_')
        if name in replace_dict:
            final_name = replace_dict[name]
            ingrs.append(final_name)
            name1 = final_name.replace('_',' ')
            if args.lower:
                ingr_name = ingr_name.lower()
                name1 = name1.lower()
            insts = insts.replace(ingr_name, final_name)
            insts = insts.replace(name1, final_name)
            title = title.replace(ingr_name, final_name)
            title = title.replace(name1, final_name)
            n += 1
    
    if n==0:
        print('no ingredients, discard')
        continue
    cvg = n/N
    cvgs.append(cvg)

    rcp['title'] = title
    rcp['ingredients'] = ingrs
    rcp['instructions'] = insts.split('\n')
    recipes.append(rcp)
    if 'images' in rcp and len(rcp['images'])>0:
        recipes_withImage.append(rcp)

cvgs = np.array(cvgs)
print('cvg = {:.2f} -- {:.2f}'.format(cvgs.mean(), cvgs.std()))
print(len(recipes), len(recipes_withImage))

print('saving...')
if args.lower and not args.remove_numbers:
    filename = 'recipes_lower'
elif not args.lower and args.remove_numbers:
    filename = 'recipes_noNumbers'
elif args.remove_numbers and args.lower:
    filename = 'recipes_lower_noNumbers'
else:
    filename = 'recipes'

with open(os.path.join(data_dir, '{}.json'.format(filename)), 'w') as f:
    json.dump(recipes, f, indent=2)

with open(os.path.join(data_dir, '{}_withImage.json'.format(filename)), 'w') as f:
    json.dump(recipes_withImage, f, indent=2)