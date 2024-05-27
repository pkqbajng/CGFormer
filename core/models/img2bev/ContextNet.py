from mmdet3d.models.builder import NECKS
from core.utils.gaussian import generate_guassian_depth_target
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import torch
import torch.nn as nn
from .modules.Mono_DepthNet_modules import ContextNet
import pdb

@NECKS.register_module()
class Context_Net(BaseModule):
    def __init__(
        self,
        downsample=8,
        numC_input=512,
        numC_Trans=64,
        cam_channels=27,
        grid_config=None,
        loss_depth_weight=1.0,
        loss_depth_type='bce',
    ):
        super(Context_Net, self).__init__()

        self.downsample = downsample
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.cam_channels = cam_channels
        self.grid_config = grid_config

        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.cam_depth_range = self.grid_config['dbound']

        self.depth_net = ContextNet(self.numC_input, self.numC_input,
                                  self.numC_Trans, self.D, cam_channels=self.cam_channels)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
            
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
    
    def forward(self, input, img_metas):
        x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input = input
        
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        img_feat = self.depth_net(x, mlp_input)
        img_feat = img_feat.view(B, N, -1, H, W)


        return img_feat, None