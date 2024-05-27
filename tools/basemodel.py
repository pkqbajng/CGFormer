import torch
import pytorch_lightning as pl

class LightningBaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def configure_optimizers(self):
        if self.config['optimizer']['type'] == 'AdamW':

            params_to_optimize = [param for param in self.model.parameters() if param.requires_grad]

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )

        else:
            raise NotImplementedError
        
        if self.config['lr_scheduler']['type'] == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['lr_scheduler']['max_lr'],
                total_steps=self.config['lr_scheduler']['total_steps'],
                pct_start=self.config['lr_scheduler']['pct_start'],
                cycle_momentum=self.config['lr_scheduler']['cycle_momentum'],
                anneal_strategy=self.config['lr_scheduler']['anneal_strategy'])

            interval=self.config['lr_scheduler']['interval']
            frequency=self.config['lr_scheduler']['frequency']
        else:
            raise NotImplementedError
        
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': interval,
            'frequency': frequency
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    