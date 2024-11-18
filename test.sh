CUDA_VISIBLE_DEVICES=0 \
nohup python main.py \
--eval \
--ckpt_path ./logs/CGFormer-Efficient-Swin-KITTI360/tensorboard/version_0/checkpoints/best.ckpt \
--config_path configs/CGFormer-Efficient-Swin-KITTI360.py \
--log_folder version1 \
--seed 7240 \
--log_every_n_steps 100 \
> CGFormer-Efficient-Swin-SemanticKITTI-Eval.log 2>&1 &