import os
import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES
from .learning_map import learning_map

@PIPELINES.register_module()
class CreateDepthFromLiDAR(object):
    def __init__(self,
        data_root=None,
        dataset='kitti',
        load_seg=False
    ):
        self.data_root = data_root
        self.dataset = dataset
        assert self.dataset in ['kitti', 'kitti360']
        if load_seg:
            self.learning_map = learning_map[dataset]
        self.seg_label_root = os.path.join(data_root, 'lidarseg')
        self.load_seg = load_seg

    def __call__(self, results):
        if self.dataset == 'kitti':
            img_filename = results['img_filename'][0]
            seq_id, _, filename = img_filename.split("/")[-3:]
            lidar_filename = os.path.join(self.data_root, 'sequences',
                            seq_id, "velodyne", filename.replace(".png", ".bin"))
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
            
            if self.load_seg:
                label_filename = os.path.join(self.seg_label_root, seq_id, 'labels', filename.replace('.png', '.label'))
                label_points = np.fromfile(label_filename, dtype=np.uint32).reshape(-1, 1, 1)
                label_points = label_points & 0xFFFF 
                label_points = np.vectorize(self.learning_map.__getitem__)(label_points)
                label_points = label_points.astype(np.uint8)
                label_points = torch.from_numpy(label_points)
            else:
                label_points = None
            
        elif self.dataset == 'kitti360':
            img_filename = results['img_filename'][0]
            seq_id, _, _, filename = img_filename.split("/")[-4:]
            lidar_filename = os.path.join(self.data_root, 'data_2d_raw', seq_id, 'velodyne_points', 'data', filename.replace(".png", ".bin"))
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
            label_points = None
        else:
            raise NotImplementedError

        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]

        # [num_point, num_img, 3] in format [u, v, d]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)
        
        if label_points is not None:
            num_point, num_img, _ = projected_points.shape
            label_points = label_points.repeat(1, num_img, 1)
            projected_points = torch.cat([projected_points, label_points], dim=-1)
        
        img_h, img_w = imgs.shape[-2:]
        valid_mask = (projected_points[..., 0] >= 0) & \
                    (projected_points[..., 1] >= 0) & \
                    (projected_points[..., 0] <= img_w - 1) & \
                    (projected_points[..., 1] <= img_h - 1) & \
                    (projected_points[..., 2] > 0)
        
        gt_depths = []
        gt_semantics = []
        for img_index in range(imgs.shape[0]):
            gt_depth = torch.zeros((img_h, img_w))
            projected_points_i = projected_points[:, img_index]
            valid_mask_i = valid_mask[:, img_index]
            valid_points_i = projected_points_i[valid_mask_i]
            # sort
            depth_order = torch.argsort(valid_points_i[:, 2], descending=True)
            valid_points_i = valid_points_i[depth_order]
            # fill in
            gt_depth[valid_points_i[:, 1].round().long(), 
                     valid_points_i[:, 0].round().long()] = valid_points_i[:, 2]
            if label_points is not None:
                gt_semantic = torch.zeros((img_h, img_w))
                gt_semantic[valid_points_i[:, 1].round().long(), 
                            valid_points_i[:, 0].round().long()] = valid_points_i[:, 3]
                gt_semantics.append(gt_semantic)

            gt_depths.append(gt_depth)

                
        gt_depths = torch.stack(gt_depths)
        results['gt_depths'] = gt_depths
        if label_points is not None:
            gt_semantics = torch.stack(gt_semantics)
            results['gt_semantics'] = gt_semantics
        
        return results

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        # from lidar to camera
        points = points.view(-1, 1, 3) # N, 1, 3
        points = points - trans.view(1, -1, 3) # N, b, 3
        inv_rots = rots.inverse().unsqueeze(0) # 1, b, 3, 3
        points = (inv_rots @ points.unsqueeze(-1)) # N, b, 3, 1
        # the intrinsic matrix is [4, 4] for kitti and [3, 3] for nuscenes 
        if intrins.shape[-1] == 4:
            points = torch.cat((points, torch.ones((points.shape[0], points.shape[1], 1, 1))), dim=2) # N, b, 4, 1
            points = (intrins.unsqueeze(0) @ points).squeeze(-1) # N, b, 4
        else:
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)

        points_d = points[..., 2:3] # N, b, 1
        points_uv = points[..., :2] / points_d # N, b, 2

        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd