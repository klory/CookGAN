import argparse
import sys
sys.path.append('/data/CS470_HnC/')
from common import root

def get_parser():
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--word2vec_dim', default=300, type=int)
    parser.add_argument('--rnn_hid_dim', default=300, type=int)
    parser.add_argument('--feature_dim', default=1024, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batches', default=400_000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--classes_file', default=f'{root}/data/Recipe1M/classes1M.pkl')
    parser.add_argument('--img_dir', default=f'{root}/data/Recipe1M/images')

    parser.add_argument('--retrieved_type', default='recipe', choices=['recipe', 'image'])
    parser.add_argument('--retrieved_range', default=1000, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--ckpt_path', default='')

    parser.add_argument('--loss_type', default='hardmining+hinge', choices=['hinge', 'hardmining+hinge', 'dynamic_soft_margin'])
    # TODO: train on other modalities
    parser.add_argument('--text_info', default='010', choices=['111', '010', '100', '001'], 
        help='3-bit to represent [title, ingredients, instructions]')
    parser.add_argument('--word2vec_file', default=f'{root}/retrieval_model/models/word2vec_recipes.bin')
    parser.add_argument('--recipe_file', default=f'{root}/data/Recipe1M/recipes_withImage.json')
    parser.add_argument('--ingrs_enc_type', default='rnn', choices=['rnn', 'fc'])
    # upmc
    parser.add_argument('--upmc_model', default=f'')
    # permute ingredients
    parser.add_argument("--permute_ingrs", type=int, default=0, choices=[0,1], help="permute ingredients")
    # self attention on text
    parser.add_argument("--with_attention", type=int, default=2, choices=[0,1,2])

    # in debug mode
    parser.add_argument("--debug", type=int, default=0, choices=[0,1], help="in debug mode or not")

    # val_retrieval.py
    parser.add_argument('--ckpt_dir', default='')

    # These are only for predict_key_ingredients.py
    parser.add_argument('--food_type', type=str, default='salad')
    parser.add_argument('--key_ingr', type=str, default='tomato')
    return parser