python3 train_retrieval_nobak.py \
--recipe_file='/data/CS470_HnC/data/Recipe1M/recipes_withImage.json' \
--img_dir='/data/CS470_HnC/data/Recipe1M/images/' \
--word2vec_file='/data/CS470_HnC/retrieval_model/models/word2vec_recipes.bin' \
--data_dir='/data/CS470_HnC/retrieval_model/models/' \
--text_info='010' \
--with_attention=2 \
--loss_type='hardmining+hinge' \
--batch_size=64