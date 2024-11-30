from .dataset import KITTIDataset, KITTI360Dataset

__datasets__ = {
    "kitti": KITTIDataset,
    "kitti360": KITTI360Dataset,
}
