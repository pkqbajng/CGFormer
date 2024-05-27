import torch
import torch.nn as nn
import torch.nn.functional as F
from .Mono_DepthNet_modules import Mlp, SELayer
from mmcv.cnn import build_norm_layer, build_conv_layer, ConvModule, build_upsample_layer
from .NeighborhoodAttention import NeighborhoodCrossAttention2D

norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)

class Attention(nn.Module):
    def __init__(self, embed_dims, kernel_size=5):
        super(Attention, self).__init__()
        self.neighbor_atttention = NeighborhoodCrossAttention2D(
            dim=embed_dims, num_heads=1, kernel_size=kernel_size, bias=True, qkv_bias=True
        )
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, q, kv):
        """
        q: b, c, h, w
        k: b, c, h, w
        """
        q = q.permute(0, 2, 3, 1)
        kv = kv.permute(0, 2, 3, 1)
        proj_value = self.neighbor_atttention(q, kv)
        out = self.gamma * proj_value + kv
        return out.permute(0, 3, 1, 2)


def convbn_2d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=pad, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1]
    )

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=pad, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1] 
    )

class StereoFeatNet(nn.Module):
    def __init__(self,
        in_channels, mid_channels, depth_channels, cam_channels
        ):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
      
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(cam_channels)
        
        self.feat_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.feat_se = SELayer(mid_channels)  # NOTE: add camera-aware
        
        self.feat_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1])) 
        x = self.reduce_conv(x) 
        feat_se = self.feat_mlp(mlp_input)[..., None, None]   
        feat = self.feat_se(x, feat_se)
        feat = self.feat_conv(feat)
        return feat

class SimpleUnet(nn.Module):
    def __init__(self, in_channels):
        super(SimpleUnet, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            convbn_2d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.redir1 = convbn_2d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class SimpleUnet3D(nn.Module):
    def __init__(self, in_channels):
        super(SimpleUnet3D, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(
            convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6

class CostVolumeEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, cam_channels, dbound, downsample):
        super(CostVolumeEncoder, self).__init__()
        self.stereo_feat_net = StereoFeatNet(in_channels, mid_channels, context_channels, cam_channels)

        self.ds = nn.Parameter(torch.arange(*dbound, dtype=torch.float32).view(1, -1, 1, 1), requires_grad=False)
        # self.downsample = downsample * 2
        self.downsample = downsample

        D = self.ds.shape[1]
        
        self.UNet = nn.Sequential(
            SimpleUnet(D),
            SimpleUnet(D)
        )
        
        self.conv_out = nn.Conv2d(D, D, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, mlp_input, calib):
        assert x.shape[1] == 2
        b, n, c, h, w = x.shape

        feat = x.view(b * n, c, h, w)
        feat = self.stereo_feat_net(feat, mlp_input)

        _, c2, h2, w2 = feat.shape
        feat = feat.view(b, n, c2, h2, w2)
        left_feat = feat[:, 0, ...]
        right_feat = feat[:, 1, ...]

        b, c, h, w = left_feat.shape

        offset = (calib / self.downsample) / self.ds.repeat(b, 1, 1, 1)
        D = offset.shape[1]

        x_grid, y_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        x_grid = x_grid.unsqueeze(0).unsqueeze(1).expand(b, D, -1, -1).to(offset.device)
        y_grid = y_grid.unsqueeze(0).unsqueeze(1).expand(b, D, -1, -1).to(offset.device)

        left_y_grid = y_grid - offset

        left_mask = (left_y_grid >= 0) & (left_y_grid <= w - 1)
        left_mask = left_mask.int().float()

        left_grid = torch.cat([left_y_grid.unsqueeze(-1), x_grid.unsqueeze(-1)], dim=-1)
        left_grid_norm = 2.0 * left_grid / torch.tensor([w - 1, h - 1], dtype=left_grid.dtype, device=offset.device) - 1.0
        sample_left = F.grid_sample(right_feat, left_grid_norm.view(b, -1, 1, 2), align_corners=True).view(b, c, D, h, w)

        cost_volume = torch.sum(left_feat.unsqueeze(2) * sample_left, dim=1)
        cost_volume = cost_volume * left_mask

        cost = self.UNet(cost_volume)

        out = self.conv_out(cost)
        return out

class ChannelAttention3D(nn.Module):
    def __init__(self, embed_dims):
        super(ChannelAttention3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, embed_dims),
            )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims//8, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
            nn.Conv3d(embed_dims//8, embed_dims, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
  
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv3d(embed_dims, embed_dims, kernel_size=3, stride=1, dilation=1, padding=1, groups=1),
            nn.GELU(),
            build_norm_layer(norm_cfg, embed_dims)[1]
        )

        self.layer_scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, input):
        x = self.conv1(input)
        y = self.sigmoid(self.conv2(self.avg_pool(x)))
        out = y * x
        out = self.conv(out)

        output_feat = input + self.layer_scale * out
        return output_feat

class DepthAggregation(nn.Module):
    def __init__(self, 
        embed_dims=32,
        out_channels=1):
        super(DepthAggregation, self).__init__()

        self.stem = nn.Conv3d(2, embed_dims, kernel_size=3, stride=1, padding=1)
        self.mono_stereo_attention = Attention(112, kernel_size=5)
        self.stereo_mono_attention = Attention(112, kernel_size=5)
        self.UNet_3D = SimpleUnet3D(in_channels=embed_dims)
        self.channel_attention = ChannelAttention3D(embed_dims=embed_dims)
        self.out_conv = nn.Conv3d(embed_dims, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, depth_stereo, depth_mono):
        mono_stereo = self.mono_stereo_attention(depth_mono, depth_stereo)
        stereo_mono = self.stereo_mono_attention(depth_stereo, depth_mono)
        
        depth_feat = torch.cat([mono_stereo.unsqueeze(1), stereo_mono.unsqueeze(1)], dim=1)
        # depth_feat = torch.cat([mono_stereo, stereo_mono], dim=1)

        depth_feat = F.relu(self.stem(depth_feat))
        depth_feat = self.UNet_3D(depth_feat)
        depth_feat = self.channel_attention(depth_feat)

        # depth_prob = F.relu(self.out_conv(depth_feat)).squeeze(1)
        depth_prob = self.out_conv(depth_feat).squeeze(1)
        return depth_prob

class DepthAggregation_wo_neighbor(nn.Module):
    def __init__(self, 
        embed_dims=32,
        out_channels=1):
        super(DepthAggregation_wo_neighbor, self).__init__()

        self.stem = nn.Conv3d(2, embed_dims, kernel_size=3, stride=1, padding=1)
        # self.mono_stereo_attention = Attention(112, kernel_size=5)
        # self.stereo_mono_attention = Attention(112, kernel_size=5)
        self.UNet_3D = SimpleUnet3D(in_channels=embed_dims)
        self.channel_attention = ChannelAttention3D(embed_dims=embed_dims)
        self.out_conv = nn.Conv3d(embed_dims, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, depth_stereo, depth_mono):
        mono_stereo = depth_mono
        # self.mono_stereo_attention(depth_mono, depth_stereo)
        stereo_mono = depth_stereo
        # self.stereo_mono_attention(depth_stereo, depth_mono)
        
        depth_feat = torch.cat([mono_stereo.unsqueeze(1), stereo_mono.unsqueeze(1)], dim=1)
        # depth_feat = torch.cat([mono_stereo, stereo_mono], dim=1)

        depth_feat = F.relu(self.stem(depth_feat))
        depth_feat = self.UNet_3D(depth_feat)
        depth_feat = self.channel_attention(depth_feat)

        # depth_prob = F.relu(self.out_conv(depth_feat)).squeeze(1)
        depth_prob = self.out_conv(depth_feat).squeeze(1)
        return depth_prob