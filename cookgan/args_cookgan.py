import argparse
import sys
sys.path.append('/data/CS470_HnC/')
from common import root

def get_parser():
    parser = argparse.ArgumentParser(description='Train a GAN network')

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--num_batches', type=int, default=200_000)

    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--base_size', type=int, default=64)

    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=100)

    parser.add_argument('--labels', type=str, default='original', choices=['original', 'R-smooth', 'R-flip', 'R-flip-smooth'])
    parser.add_argument("--input_noise", type=int, default=0)
    parser.add_argument('--uncond', type=float, default=1.0)
    parser.add_argument('--cycle_txt', type=float, default=0.0)
    parser.add_argument('--cycle_img', type=float, default=0.0)
    # parser.add_argument('--tri_loss', type=float, default=0.0)
    parser.add_argument('--kl', type=float, default=2.0)

    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)


    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--food_type', type=str, default='salad')

    parser.add_argument('--recipe_file', type=str, default=f'{root}/data/Recipe1M/recipes_withImage.json')
    parser.add_argument('--img_dir', type=str, default=f'{root}/data/Recipe1M/images')
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--retrieval_model', type=str, default=f'{root}/retrieval_model/wandb/run-20201204_174135-6w1fft7l/files/00000000.ckpt')
    parser.add_argument('--word2vec_file', type=str, default=f'{root}/retrieval_model/models/word2vec_recipes.bin')

    parser.add_argument("--debug", type=int, default=0)

    # These are only for test_StackGAN.py
    parser.add_argument('--level', type=int, default=2)

    # These are only for interpolate.py
    parser.add_argument('--key_ingr', type=str, default='tomato')
    return parser