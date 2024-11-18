import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder

@DETECTORS.register_module()
class CGFormerSegDepth(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        plugin_head=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None
        ):
        super().__init__()
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.plugin_head = builder.build_head(plugin_head)
        # self.img_view_transformer = builder.build_neck(img_view_transformer)
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs) 

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        img = img_inputs[0]
        img_enc_feats = self.image_encoder(img)
        B, N, _, _, _ = img_enc_feats.shape

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas)

        if len(context.shape) == 5:
            b, n, d, h, w = context.shape
            context = context.view(b * n, d, h, w)
        
        return context, depth
    
    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        # gt_occ = data_dict['gt_occ']
        target = data_dict['gt_semantics']

        context, depth = self.extract_img_feat(img_inputs=img_inputs, img_metas=img_metas)

        segmentation = self.plugin_head(context)

        losses = dict()
        losses['loss_depth'] = self.depth_net.get_depth_loss(img_metas['gt_depths'], depth)

        losses_seg = self.plugin_head.loss(
            pred=segmentation,
            target=target[:, 0:1, ...],
            depth=img_metas['gt_depths'][:, 0:1, ...]
        )
        losses.update(losses_seg)

        train_output = {
            'losses': losses
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']

        context, depth = self.extract_img_feat(img_inputs=img_inputs, img_metas=img_metas)

        segmentation = self.plugin_head(context)
        
        test_output = {
            'pred': segmentation,
            'depth': depth
        }
        return test_output
    
    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None