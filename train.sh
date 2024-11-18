CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python main.py \
--config_path configs/CGFormer-Efficient-Swin-SemanticKITTI-Pretrain.py \
--log_folder CGFormer-Efficient-Swin-SemanticKITTI-Pretrain \
--seed 7240 \
--pretrain \
--log_every_n_steps 100 \
> CGFormer-Efficient-Swin-SemanticKITTI-Pretrain.log 2>&1 &