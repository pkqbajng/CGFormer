import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from LightningTools.dataset_dm import DataModule
from mmcv import Config
from mmdet3d_plugin import *
from argparse import ArgumentParser
from mmdet3d.models import build_model
from mmdet.datasets import build_dataset
from torch.utils.data.dataloader import DataLoader
from LightningTools.pl_model import pl_model

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/CGFormer-Efficient-Swin-SemanticKITTI-Pretrain.py')
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

def tensor_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [tensor_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: tensor_to_device(value, device) for key, value in data.items()}
    else:
        return data

if __name__ == '__main__':
    args, config = parse_config()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    train_dataset = build_dataset(config.data.train)
    data = train_dataset[0]
    data_dm = DataModule(config)
    data_dm.setup()
    # # model = pl_model(config)
    train_dataloader = data_dm.train_dataloader()

    model = build_model(config.model)
    model = model.to(device)
    for i, val in enumerate(train_dataloader):
        val = tensor_to_device(val, device)
        output = model(val)