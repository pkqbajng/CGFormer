import os
import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES

learning_map={
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

@PIPELINES.register_module()
class CreateDepthAndSemanticFromLiDAR(object):

    def __init__(self,
        data_root=None,
        seg_label_root=None,
        dataset='kitti'
    ):
        self.data_root = data_root
        self.seg_label_root = seg_label_root
        self.dataset = dataset
        self.learning_map = learning_map
        assert self.dataset in ['kitti']

    def __call__(self, results):
        if self.dataset == 'kitti':
            img_filename = results['img_filename'][0]
            seq_id, _, filename = img_filename.split("/")[-3:]
            lidar_filename = os.path.join(self.data_root, 'sequences',
                            seq_id, "velodyne", filename.replace(".png", ".bin"))
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
            label_filename = os.path.join(self.seg_label_root, seq_id, 'labels', filename.replace('.png', '.label'))
            label_points = np.fromfile(label_filename, dtype=np.uint32).reshape(-1, 1, 1)
            label_points = label_points & 0xFFFF 
            label_points = np.vectorize(self.learning_map.__getitem__)(label_points)
            label_points = label_points.astype(np.uint8)
            label_points = torch.from_numpy(label_points)

        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]

        # [num_point, num_img, 3] in format [u, v, d]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)
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
            gt_semantic = torch.zeros((img_h, img_w))
            projected_points_i = projected_points[:, img_index]
            valid_mask_i = valid_mask[:, img_index]
            valid_points_i = projected_points_i[valid_mask_i]
            # sort
            depth_order = torch.argsort(valid_points_i[:, 2], descending=True)
            valid_points_i = valid_points_i[depth_order]
            # fill in
            gt_depth[valid_points_i[:, 1].round().long(), 
                     valid_points_i[:, 0].round().long()] = valid_points_i[:, 2]
            gt_semantic[valid_points_i[:, 1].round().long(), 
                     valid_points_i[:, 0].round().long()] = valid_points_i[:, 3]
            
            gt_depths.append(gt_depth)
            gt_semantics.append(gt_semantic)

                
        gt_depths = torch.stack(gt_depths)
        gt_semantics = torch.stack(gt_semantics)
        imgs, rots, trans, intrins, post_rots, post_trans, _, sensor2sensors, focal_length, baseline = results['img_inputs']
        results['img_inputs'] = imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors, focal_length, baseline
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