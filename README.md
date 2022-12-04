# Made A Little CookGAN

#### Team 8 (Yeon Su Park, Jinsuk Kim, Yoojin Hong, Honggi Lee)
The existing image generation models cannot synthesize photo-realistic meal images which contextualize the amount of each ingredient used in a recipe.<br/>
We tackled the challenge of including contextual information when generating realistic meal images by automatically adjusting the amount of ingredients in the image generation process through latent space interpolation.

## Environmental Setting
- **requirements.txt:** environment of Python 3.8 & required packages
- **environment.yaml:** conda environment named ys2

## Prepare Dataset
- Download Recipe1M dataset from http://pic2recipe.csail.mit.edu/ and place it inside CS470_HnC/data/Recipe1M/. <br/>
- ```
  CS470_HnC/data/Recipe1M/
    images/
        train/
        val/
        test/
    recipe1M/
        det_ingrs.json
        layer1.json
        layer2.json
- run `python clean_recipes_with_canonical_ingrs.py` to generate `./data/Recipe1M/recipes_withImage.json` which contains *simplified* recipes with images (N=402760).

## Train Ingredient Encoder
- **CS470_HnC/retrieval_model/train_word2vec.py:** Train Word2Vec to Generate `models/word2vec_recipes.bin`.

## Train Image Encoder
- Download UPMC-Food-101 dataset from [HERE](https://drive.google.com/drive/folders/1cpb5g0I5DJAffqEaJ3gLKiySJ8KGopPN) and place it inside CS470_HnC/retrieval_model/.pretrain_upmc/. <br/>
- **CS470_HnC/retrieval_model/pretrain_upmc/train_upmc.py:** Train Image Encoder on UPMC-Food-101 dataset.
- The training process can be viewed [HERE](https://wandb.ai/hnc/cookgan_pretrain_upmc?workspace=user-yeonsuuuu28).

## Train FoodSpace
- **CS470_HnC/retrieval_model/run_retrieval_nobak.sh:** Train Attention-based Retrieval Model.
- The training process can be viewed [HERE](https://wandb.ai/hnc/cookgan_retrieval_model?workspace=user-yeonsuuuu28).

## Train CookGAN
- **CS470_HnC/cookgan/run.sh:** Train CookGAN on salad.
- The training process can be viewed [HERE](https://wandb.ai/hnc/cookgan?workspace=user-yeonsuuuu28).

## Conduct Interpolation In Latent Space
- **CS470_HnC/made_a_little_cookgan/run_interpolation.ipynb:** Generate Meal Image with Ingredient List & Conduct Appropriate Interpolation. 
- **CS470_HnC/made_a_little_cookgan/interpolation_example/:** Example Interpolation Results. See [this](https://github.com/alexhonggi/CS470_HnC/blob/main/made_a_little_cookgan/interpolation_save/tomato_interpolations.jpg).
- The output can be previewed from the [`run_interpolation.ipynb` jupyter notebook](https://github.com/alexhonggi/CS470_HnC/blob/main/made_a_little_cookgan/run_interpolation.ipynb). The step-by-step instruction is given in the file itself.
