import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from mmdet.datasets import build_dataset
from torch.utils.data.dataloader import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config      
    ):
        super().__init__()
        self.trainset_config = config.data.train
        self.testset_config = config.data.test
        self.valset_config = config.data.val

        self.train_dataloader_config = config.train_dataloader_config
        self.test_dataloader_config = config.test_dataloader_config
        self.val_dataloader_config = config.test_dataloader_config
        self.config = config
    
    def setup(self, stage=None):
        self.train_dataset = build_dataset(self.trainset_config)
        self.test_dataset = build_dataset(self.testset_config)
        self.val_dataset = build_dataset(self.valset_config)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataloader_config.batch_size,
            drop_last=True,
            num_workers=self.train_dataloader_config.num_workers,
            shuffle=True,
            pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.val_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.test_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=True)