import sys
sys.path.append('../')
from common import load_recipes
from tqdm import tqdm
import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import argparse
import pdb

parser = argparse.ArgumentParser(description='train word2vec model')
parser.add_argument('--recipe_file', default='../data/Recipe1M/recipes.json')
args = parser.parse_args()

print('Load documents...')
file_path = args.recipe_file
recipes = load_recipes(file_path, 'train')

print('Tokenize...')
all_sentences = []
for entry in tqdm(recipes):
    all_sentences.append(entry['title'].split())
    insts = entry['instructions']
    sentences = [inst.split() for inst in insts]
    all_sentences.extend(sentences)
    all_sentences.append(entry['ingredients'])
print('number of sentences =', len(all_sentences))

print('Train Word2Vec model...')
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print('-' * 40)
        print("Epoch #{} start".format(self.epoch))
        print('vocab_size = {}'.format(len(model.wv.index2word)))
    def on_epoch_end(self, model):
        print('total_train_time = {:.2f} s'.format(model.total_train_time))
        print('loss = {:.2f}'.format(model.get_latest_training_loss()))
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

epoch_logger = EpochLogger()
model = Word2Vec(
    all_sentences, size=300, window=10, min_count=10, 
    workers=20, iter=10, callbacks=[epoch_logger], 
    compute_loss=True)

suffix = os.path.basename(file_path).split('.')[0]
if not os.path.exists('models'):
    os.makedirs('models')

model.wv.save(os.path.join('models/word2vec_{}.bin'.format(suffix)))
