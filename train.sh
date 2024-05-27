CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--config_path configs/semantickitti_CGFormer.py \
--log_folder semantickitti_CGFormer \
--seed 7240 \
--log_every_n_steps 100 
