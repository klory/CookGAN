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
- Download UPMC-Food-101 dataset from [HERE] and place it inside CS470_HnC/retrieval_model/pretrain_upmc/. <br/>
- **CS470_HnC/retrieval_model/pretrain_upmc/train_upmc.py:** Train Image Encoder on UPMC-Food-101 dataset.

## Train FoodSpace
- **CS470_HnC/retrieval_model/run_retrieval_nobak.sh:** Train Attention-based Retrieval Model.

## Train CookGAN
- **CS470_HnC/cookgan/run.sh:** Train CookGAN on salad.

## Conduct Interpolation
- **CS470_HnC/made_a_little_cookgan/:** Contain Our Code Files for Interpolation
- **CS470_HnC/made_a_little_cookgan/run_interpolation.ipynb:** Generate Meal Image with Ingredient List & Conduct Appropriate Interpolation
- **CS470_HnC/made_a_little_cookgan/interpolation_example/:** Example Interpolation Results
- The output can be previewed from the `run_interpolation.ipynb` jupyter notebook. 
