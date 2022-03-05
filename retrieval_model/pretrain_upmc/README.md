## Prepare dataset
Download UPMC-Food-101 from http://visiir.lip6.fr/

> 2021-06-16: UPMC-Food-101 has changed to a newer version, the one we used is [HERE](https://drive.google.com/drive/folders/1URwnLMVKx3avmUI0ITjxzgjFkpvmiS3Q?usp=sharing).

## Train
```
CUDA_VISIBLE_DEVICES=0 python train_upmc.py --data_dir=your/data_dir --batch_size=128
```

# Note
The code is tested on one Tesla K80 with 12Gb memory (batch_size=128 takes about 12 Gb memory). You could also run it with multiple GPUs by:
```
CUDA_VISIBLE_DEVICES=0,1 python train_upmc.py --data_dir=your/data_dir --batch_size=256
```

