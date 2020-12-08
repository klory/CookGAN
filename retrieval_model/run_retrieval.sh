python train_retrieval.py \
--recipe_file='/common/home/fh199/CookGAN/data/Recipe1M/original_withImage.json' \
--img_dir='/common/home/fh199/CookGAN/data/Recipe1M/images/' \
--word2vec_file='/common/home/fh199/CookGAN/retrieval_model/models/word2vec_recipes.bin' \
--text_info='010' \
--with_attention=2 \
--loss_type='hardmining+hinge' \
--batch_size=64