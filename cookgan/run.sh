python train_cookgan.py \
--recipe_file='/common/home/fh199/CookGAN/data/Recipe1M/recipes_withImage.json' \
--img_dir='/common/home/fh199/CookGAN/data/Recipe1M/images/' \
--retrieval_model='/common/home/fh199/CookGAN/retrieval_model/wandb/run-20201204_174135-6w1fft7l/files/00000000.ckpt' \
--levels=3 \
--food_type='salad' \
--base_size=64 \
--batch_size=16 \
--workers=16