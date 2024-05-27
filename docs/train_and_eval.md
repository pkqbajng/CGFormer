## Training Details

We train CGFormer for 25 epochs on 4 NVIDIA 4090 GPUs, with a batch size of 4. It approximately consumes 19GB of GPU memory on each GPU during the training phase. Before start training, download the corresponding pretrained checkpoints from this [link](https://drive.google.com/drive/folders/1caNRjcGyBi6iUfQgGxMJB5SH69b5M90h?usp=sharing) and put them under the folder pretrain.

## Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer \
--seed 7240 \
--log_every_n_steps 100
```

The training logs and checkpoints will be saved under the log_folder

## Evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval --seed 7240 \
--log_every_n_steps 100
```

## Evaluation with Saving the Results

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred
```

The results will be saved into the save_path.

## Submission

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --test_mapping
```