python train_cookgan.py \
--recipe_file='/data/CS470_HnC/data/Recipe1M/recipes_withImage.json' \
--img_dir='/data/CS470_HnC/data/Recipe1M/images' \
--retrieval_model='/data/CS470_HnC/retrieval_model/wandb/run-20221115_141017-qn8zgvm8/files/00390000.ckpt' \
--levels=3 \
--food_type='salad' \
--base_size=64 \
--batch_size=16 \
--workers=16