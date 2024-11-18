# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmcv.runner import BaseModule

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@NECKS.register_module()
class LSSViewTransformer(BaseModule):
    def __init__(
        self, grid_config=None, data_config=None, downsample=8,
    ):
        super().__init__()

        self.grid_config = grid_config
        self.data_config = data_config
        self.downsample = downsample
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] == [b, c, x, y, z]
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final
    
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points

    
    def forward(self, feat, depth_prob, cam_params):
        B, N, C, H, W = feat.shape
        rots, trans, intrins, post_rots, post_trans, bda = cam_params

        if len(depth_prob.shape) == 4:
            db, cb, dh, dw = depth_prob.shape
            assert db == B * N
            depth_prob = depth_prob.view(B, N, cb, dh, dw)
        # Lift
        volume = depth_prob.unsqueeze(2) * feat.unsqueeze(3)
        volume = volume.view(B, N, -1, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        bev_feat = self.voxel_pooling(geom, volume)

        return bev_feat
