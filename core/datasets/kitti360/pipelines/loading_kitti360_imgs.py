import mmcv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_KITTI360(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """
    def __init__(
        self,
        data_config,
        is_train=False,
        img_norm_cfg=None,
        load_stereo_depth=False,
        color_jitter=(0.4, 0.4, 0.4)
    ):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config
        self.img_norm_cfg = img_norm_cfg

        self.load_stereo_depth = load_stereo_depth
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.normalize_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.ToTensor = transforms.ToTensor()
    
    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
    
    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])

        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0

        return resize, resize_dims, crop, flip, rotate
    
    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img
    
    def get_inputs(self, results, flip=None, scale=None):
        img_filenames = results['img_filename']

        focal_length = results['focal_length']
        baseline = results['baseline']

        data_lists = []
        raw_img_list = []

        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            img = Image.open(img_filename).convert('RGB')

            # perform image-view augmentation
            post_rot = torch.eye(2)
            post_trans = torch.zeros(2)

            if i == 0:
                img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_trans, resize=resize, 
                resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # intrins
            intrin = torch.Tensor(results['cam_intrinsic'][i])

            # extrins
            lidar2cam = torch.Tensor(results['lidar2cam'][i])
            cam2lidar = lidar2cam.inverse()
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]

            # output
            canvas = np.array(img)

            if self.color_jitter and self.is_train:
                img = self.color_jitter(img)
            
            img = self.normalize_img(img)
            depth = torch.zeros(1)

            result = [img, rot, tran, intrin, post_rot, post_tran, depth, cam2lidar]
            result = [x[None] for x in result]

            data_lists.append(result)
            raw_img_list.append(canvas)
        
        if self.load_stereo_depth:
            stereo_depth_path = results['stereo_depth_path']
            stereo_depth = np.load(stereo_depth_path)
            stereo_depth = Image.fromarray(stereo_depth)
            resize, resize_dims, crop, flip, rotate = img_augs
            stereo_depth = self.img_transform_core(stereo_depth, resize_dims=resize_dims,
                    crop=crop, flip=flip, rotate=rotate)
            results['stereo_depth'] = self.ToTensor(stereo_depth)
        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))
        
        result_list.append(torch.tensor(focal_length, dtype=torch.float32))
        result_list.append(torch.tensor(baseline, dtype=torch.float32))
        results['raw_img'] = raw_img_list

        return result_list
    
    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)

        return results