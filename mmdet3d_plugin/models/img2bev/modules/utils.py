import torch
import numpy as np
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.conv_module import ConvModule

class Voxelization(nn.Module):
    def __init__(self, point_cloud_range, spatial_shape):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = np.array([
            [point_cloud_range[0], point_cloud_range[3]],
            [point_cloud_range[1], point_cloud_range[4]],
            [point_cloud_range[2], point_cloud_range[5]]
        ])

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def filter_pc(self, pc, batch_idx):
        def mask_op(data, x_min, x_max):
            mask = (data > x_min) & (data < x_max)
            return mask
        mask_x = mask_op(pc[:, 0], self.coors_range_xyz[0][0] + 0.0001, self.coors_range_xyz[0][1] - 0.0001)
        mask_y = mask_op(pc[:, 1], self.coors_range_xyz[1][0] + 0.0001, self.coors_range_xyz[1][1] - 0.0001)
        mask_z = mask_op(pc[:, 2], self.coors_range_xyz[2][0] + 0.0001, self.coors_range_xyz[2][1] - 0.0001)
        mask = mask_x & mask_y & mask_z
        filter_pc = pc[mask]
        fiter_batch_idx = batch_idx[mask]
        if filter_pc.shape[0] < 10:
            filter_pc = torch.ones((10, 3), dtype=pc.dtype).to(pc.device)
            filter_pc = filter_pc * torch.rand_like(filter_pc)
            fiter_batch_idx = torch.zeros(10, dtype=torch.long).to(pc.device)
        return filter_pc, fiter_batch_idx

    def forward(self, pc, batch_idx):
        pc, batch_idx = self.filter_pc(pc, batch_idx)
        xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], self.spatial_shape[0])
        yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], self.spatial_shape[1])
        zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], self.spatial_shape[2])

        bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
        unq, unq_inv, _ = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)

        return unq, unq_inv

class _ASPPModule(BaseModule):
    def __init__(self, 
            inplanes, 
            planes, 
            kernel_size, 
            padding, 
            dilation,
            norm_cfg=dict(type='BN'),
            conv_cfg=None,
        ):
        super(_ASPPModule, self).__init__()
        
        self.atrous_conv = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn(self.atrous_conv(x))
        x = self.relu(x)

        return x

class ASPP(BaseModule):
    def __init__(self,
            inplanes,
            mid_channels=None,
            dilations=[1, 6, 12, 18],
            norm_cfg=dict(type='BN'),
            conv_cfg=None,
            dropout=0.1,
        ):
        super(ASPP, self).__init__()
        
        if mid_channels is None:
            mid_channels = inplanes // 2
        
        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 norm_cfg=norm_cfg,
                                 conv_cfg=conv_cfg)
        
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 norm_cfg=norm_cfg,
                                 conv_cfg=conv_cfg)
        
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 norm_cfg=norm_cfg,
                                 conv_cfg=conv_cfg)
        
        # we set the output channel the same as the input
        outplanes = inplanes
        self.conv1 = build_conv_layer(conv_cfg, int(mid_channels * 3), outplanes, 1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, outplanes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weight()

    def forward(self, x):
        identity = x.clone()
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return identity + self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)