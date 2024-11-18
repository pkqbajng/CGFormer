import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss

@HEADS.register_module()
class plugin_segmentation_head(nn.Module):
    def __init__(
        self,
        empty_idx=0,
        in_channels=128,
        out_channel_list=[128, 64, 32],
        num_class=20,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        train_cfg=None,
        test_cfg=None
    ):
        super(plugin_segmentation_head, self).__init__()
        self.empty_idx = empty_idx
        in_channel = in_channels
        self.deconv_blocks = nn.ModuleList()
        for out_channel in out_channel_list:
            self.deconv_blocks.append(
                nn.Sequential(
                build_upsample_layer(
                upsample_cfg,
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=2,
                stride=2),
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True))
            )
            in_channel = out_channel
        
        self.pred = nn.Conv2d(out_channel_list[-1], num_class, kernel_size=1, stride=1)
    
    def forward(self, x):
        for deconv_block in self.deconv_blocks:
            x = deconv_block(x)
        
        seg_pred = self.pred(x)
        return seg_pred
    
    def loss(self, pred, target, depth):
        criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction="mean")

        b, d, h, w = pred.shape
        if len(target.shape) == 4:
            tb, tn, h, w = target.shape
            assert tb * tn == b
            target = target.view(-1, h, w)
        if len(depth.shape) == 4:
            tb, tn, h, w = depth.shape
            assert tb * tn == b
            depth = depth.view(-1, h, w)
        loss_value = 0
        for i in range(b):
            pred_sample = pred[i, ...]
            sample_mask = depth[i, ...] > 0
            target_sample = target[i, ...]

            pred_points = pred_sample.permute(1, 2, 0)[sample_mask].permute(1, 0)
            target_points = target_sample[sample_mask]

            loss_value += criterion(pred_points.unsqueeze(0), target_points.unsqueeze(0).long())

        loss_dict = {}
        loss_dict['loss_voxel_ce'] = loss_value / b
        return loss_dict
            

