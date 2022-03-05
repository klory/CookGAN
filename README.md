This is the official repository for the WACV paper [CookGAN: Meal Image Synthesis from Ingredients](https://openaccess.thecvf.com/content_WACV_2020/papers/Han_CookGAN_Meal_Image_Synthesis_from_Ingredients_WACV_2020_paper.pdf). The code is tested with Python 3.8, PyTorch 1.6, CUDA 10.2 on Ubuntu 18.04

# Prepare Dataset
## Dowload original dataset
Download Recipe1M dataset from http://pic2recipe.csail.mit.edu/, make sure you have download and unzip all images and the files `det_ingrs.json`, `layer1.json`, `layer2.json`. Your data folder should look like the one shown below
```
CookGAN/data/Recipe1M/
    images/
        train/
        val/
        test/
    recipe1M/
        det_ingrs.json
        layer1.json
        layer2.json
```
Create an environment using Python 3.8, install the required packages.
```
pip install -r requirements.txt
```

## Simplify dataset
run `python clean_recipes_with_canonical_ingrs.py` to generate `./data/Recipe1M/recipes_withImage.json` which just contains the *simplified* recipes with images (N=402760), e.g.
```
{
    "id": "00003a70b1",
    "url": "http://www.food.com/recipe/crunchy-onion-potato-bake-479149",
    "partition": "test",
    "title": "Crunchy Onion Potato Bake",
    "instructions": [
      "Preheat oven to 350 degrees Fahrenheit.",
      "Spray pan with non stick cooking spray.",
      "Heat milk, water and butter to boiling; stir in contents of both pouches of potatoes; let stand one minute.",
      "Stir in corn.",
      "Spoon half the potato mixture in pan.",
      "Sprinkle half each of cheese and onions; top with remaining potatoes.",
      "Sprinkle with remaining cheese and onions.",
      "Bake 10 to 15 minutes until cheese is melted.",
      "Enjoy !"
    ],
    "ingredients": [
      "milk",
      "water",
      "butter",
      "mashed potatoes",
      "whole kernel corn",
      "cheddar cheese",
      "French - fried onions"
    ],
    "valid": [
      true,
      true,
      true,
      true,
      true,
      true,
      true
    ],
    "images": [
      "3/e/2/3/3e233001e2.jpg",
      "7/f/7/4/7f749987f9.jpg",
      "a/a/f/6/aaf6b2dcd3.jpg"
    ]
  }
```

# Train Models
All models (except word2vec) could be monitored using [wandb](https://www.wandb.com/).

## Train word2vec
Go to `retrieval_model` and run `python train_word2vec.py` to generate `models/word2vec_recipes.bin`.

## Pre-train UPMC-Food-101 classifier
Go to `./pretrain_upmc`, follow `./pretrain_upmc/README` to pretrain the image encoder on UPMC-Food-101 dataset.

## Train the attention-based retrieval model
Run 
```
CUDA_VISIBLE_DEVICES=0 bash run_retrieval.sh
```
to train the [attention-based recipe retrieval model](https://dl.acm.org/citation.cfm?id=3240627). Here, `010` means only using ingredients to train the model. The code also supports training using all three domains by `--text_info=111` (title+ingredients+instructions).

## Train CookGAN
Go to `CookGAN/cookgan` and run
```
CUDA_VISIBLE_DEVICES=0 bash run.sh
```
to train CookGAN on salad.

## Test Models
go to `CookGAN/metrics/`, 

* Update the configurations following `configs/salad+cookgan.yaml`.
* Run `python calc_inception.py` to generate statistics for real images.
* Run `python fid.py` to compute the FIDs under a certain checkpoint directory.
* Run `python medR.py` to compute the median ranks under a certain checkpoint directory.

### Genearte an image from the trained model

1. Download the trained model from the [Google drive folder](https://drive.google.com/drive/folders/1URwnLMVKx3avmUI0ITjxzgjFkpvmiS3Q?usp=sharing).
2. Run the notebook test_model.ipynb to generate an image.

<!-- # Experiemnts
## Generate fake images
>assume you have `models/salad.ckpt` and `models/salad.json`

cd to `generative_model` and generate all images from val_set using your trained model~
```
CUDA_VISIBLE_DEVICES=0 python eval_StackGANv2.py --food_type=salad --resume=models/salad.ckpt
```
and see the results in `./experiments/salad/`

## Predict ingredients from 'same' fake images
>assume you have `generative_model/models/salad.ckpt`, `generative_model/models/salad.json` and `models/salad.ckpt`, `models/salad.json`

cd to `food_attention` and run
```
CUDA_VISIBLE_DEVICES=0 python same_recipe_different_noises.py \
--food_type=salad \
--retrieval_model=models/010.ckpt \
--generation_model=generative_model/models/salad.ckpt
```
It will ask you to input an integer as the index of the salad recipes, then

1. generate 64 fake images using this recipe's text info with 64 different noises, 
2. use the retrieval model to retrieve the most similar recipe text for each fake image, 
3. find the top 20 ingredients in terms of appearance

You could find see results in `./experiments/salad/`, e.g.

- `0_fake.jpg`: the 64 generated images
- `0_pred.jpeg`: the top20 ingredients
- `0_real.jpg`: the real image
- `0_report.txt`: the report file

## Interpolate between txt_feats
>assume you have `generative_model/models/salad.ckpt`, `generative_model/models/salad.json` and `models/salad.ckpt`, `models/salad.json`

cd to `food_attention` and run
```
CUDA_VISIBLE_DEVICES=0 python eval_ingr_retrieval.py \
--resume=models/010.ckpt \
--generation_model=generative_model/models/salad.ckpt \
--food_type=salad \
--hot_ingr=red_pepper \
--save_dir=experiments
```

This experiment follows the steps below:

1. find all `salad` recipes in val_set (N=3670)
2. find salad recipes with red_pepper (N=352), and without red_pepper (N=3318)
```
hot_ingr: red_pepper
#with=352/3670 = 0.10, #without=3318/3670 = 0.9
```
3. find the recipe pairs (one with red_pepper and the other without red_pepper, the rest ingredients have at least 70% overlap), e.g. #paris=6. Then remove the duplicates, #uniques_with=4, #uniques_without=6
```
#red_pepper pairs (IoU=0.70) = 6
#unique = 4, #unique_ = 6
```
4. Use REAL images from unique_with to retrieve recipe text, for each image, see how many recipes contain red_pepper out of the top 5 retrieved recipes. e.g.
```
Top 5 avg coverage with red_pepper (#=4) = 0.25 (0.22)
```
5. Repeat Step 4 for unique_without
```
Top 5 avg coverage without red_pepper (#=6) = 0.03 (0.07)
```
6. Interpolate txt_feats between with (N=4) and without (N=4, just choose the first 4 recipes) red_pepper, generate FAKE images using interpolations and repeat Step 4. e.g.
```
interpolate points: [1.0, 0.75, 0.5, 0.25, 0.0]
with/without=1.00/0.00, avg cvg (over 4 recipes)=0.40 (0.37)
with/without=0.75/0.25, avg cvg (over 4 recipes)=0.45 (0.38)
with/without=0.50/0.50, avg cvg (over 4 recipes)=0.35 (0.36)
with/without=0.25/0.75, avg cvg (over 4 recipes)=0.20 (0.00)
with/without=0.00/1.00, avg cvg (over 4 recipes)=0.05 (0.09)
```
The generated files are stored in `experiments/salad/`, e.g.

- `tomato_interopolations.jpg`: this contains eight rows of fake images with/without = [1.0, 0.75, 0.5, 0.25, 0.0] (from left to right)
- `tomato_with.jpg` : this is real images with tomato
- `tomato_with.json`: this is the recipes with tomato
- `tomato_without.jpg`: this is real images without tomato
- `tomato_without.json`: this is the recipes without tomato -->

# License
MIT