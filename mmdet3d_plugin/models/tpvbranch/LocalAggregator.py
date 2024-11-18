from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class LocalAggregator(BaseModule):
    def __init__(
        self,
        local_encoder_backbone=None,
        local_encoder_neck=None,
    ):
        super().__init__()
        self.local_encoder_backbone = builder.build_backbone(local_encoder_backbone)
        self.local_encoder_neck = builder.build_neck(local_encoder_neck)
    
    def forward(self, x):
        x_list = self.local_encoder_backbone(x)
        output = self.local_encoder_neck(x_list)
        output = output[0]

        return output