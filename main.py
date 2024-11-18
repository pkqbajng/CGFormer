import os
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()
    log_folder = os.path.join('logs', config['log_folder'])
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    model = pl_model(config)
    
    data_dm = DataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        save_last=True,
        filename='best')
    
    if not config.eval:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            max_steps=config.training_steps,
            resume_from_checkpoint=None,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step')
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch']
        )
        trainer.fit(model=model, datamodule=data_dm)
    else:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            logger=tb_logger,
            profiler=profiler
        )
        trainer.test(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])

    

