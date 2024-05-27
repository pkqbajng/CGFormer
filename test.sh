# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python main.py \
# --eval \
# --ckpt_path ./logs/semantickitti_CGFormer/tensorboard/version_0/checkpoints/best.ckpt \
# --config_path configs/semantickitti_CGFormer.py \
# --log_folder semantickitti_CGFormer_eval \
# --seed 7240 \
# --log_every_n_steps 100 

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--eval \
--ckpt_path ./logs/semantickitti_CGFormer/tensorboard/version_0/checkpoints/best.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer_eval \
--seed 7240 \
--log_every_n_steps 100 \
--save_path pred 