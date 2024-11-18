import timm
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES
import torch.utils.model_zoo as model_zoo

@BACKBONES.register_module()
class CustomResNet(BaseModule):
    def __init__(
        self,
        arch='resnet50d.a1_in1k',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        drop_path_rate=0.5,
        init_cfg=None,
        **kwargs
    ):
        super().__init__()
        if pretrained is not None:
            model = timm.create_model(arch, pretrained=True, pretrained_cfg_overlay=dict(file=pretrained), drop_path_rate=drop_path_rate)
        else:
            model = timm.create_model(arch, pretrained=False)
        self.conv1 = model.conv1
        self.norm1 = model.bn1
        self.relu = model.act1

        self.maxpool = model.maxpool

        assert max(out_indices) < num_stages
        self.out_indices = out_indices
        self.res_layers = nn.ModuleList()

        self.res_layers.append(model.layer1)
        self.res_layers.append(model.layer2)
        self.res_layers.append(model.layer3)
        self.res_layers.append(model.layer4)

        self.res_layers = self.res_layers[:num_stages]

        del model
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)