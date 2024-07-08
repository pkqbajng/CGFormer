## Training Details

We train CGFormer for 25 epochs on 4 NVIDIA 4090 GPUs, with a batch size of 4. It approximately consumes 19GB of GPU memory on each GPU during the training phase. Before start training, download the corresponding pretrained checkpoints ([efficientnet](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth), [pretrain_geodepth](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/pretrain_geodepth.pth) and [swin tiny](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/swin_tiny_patch4_window7_224.pth)) and put them under the folder pretrain.

## Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer \
--seed 7240 \
--log_every_n_steps 100
```

The training logs and checkpoints will be saved under the log_folder、

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the ckpts folder.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/CGFormer_semantickitti.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval --seed 7240 \
--log_every_n_steps 100
```

## Evaluation with Saving the Results

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/CGFormer_semantickitti.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred
```

The results will be saved into the save_path.

## Submission

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/CGFormer_semantickitti.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --test_mapping
```