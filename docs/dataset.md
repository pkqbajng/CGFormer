## SemanticKITTI

Download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Please follow the command [image2depth_semantickitti](../preprocess/image2depth_semantickitti.sh) to create depth maps and preprocess the annotations for semantic scene completion:

```bash
python tools/preprocess.py --kitti_root data/SemanticKITTI --kitti_preprocess_root data/SemanticKITTI
```

### Folder structure

The data is organized in the following format:

```
/semantickittii/
          |-- sequences/
          │       |-- 00/
          │       │   |-- poses.txt
          │       │   |-- calib.txt
          │       │   |-- image_2/
          │       │   |-- image_3/
          │       |   |-- voxels/
          │       |         |- 000000.bin
          │       |         |- 000000.label
          │       |         |- 000000.occluded
          │       |         |- 000000.invalid
          │       |         |- 000005.bin
          │       |         |- 000005.label
          │       |         |- 000005.occluded
          │       |         |- 000005.invalid
          │       |-- 01/
          │       |-- 02/
          │       .
          │       |-- 21/
          |-- labels/
          │       |-- 00/
          │       │   |-- 000000_1_1.npy
          │       │   |-- 000000_1_2.npy
          │       │   |-- 000005_1_1.npy
          │       │   |-- 000005_1_2.npy
          │       |-- 01/
          │       .
          │       |-- 10/
          |-- lidarseg/
          |       |-- 00/
          |       │   |-- labels/
          |       |         ├ 000001.label
          |       |         ├ 000002.label
          |       |-- 01/
          |       |-- 02/
          |       .
          |       |-- 21/
          |-- depth/sequences/
          		  |-- 00/
          		  │   |-- 000000.npy
          		  |   |-- 000001.npy
          		  |-- 01/
                  |-- 02/
                  .
                  |-- 21/
          
```

## SSCBench-KITTI-360

Download the dataset from [SSCBench-KITTI-360](https://github.com/ai4ce/SSCBench) and prepare the depth maps using [image2depth_kitti360](../preprocess/image2depth_kitti360.sh).

### Folder Structure

The data is organized in the following format:

```
/SSCBenchKITTI360/
    |-- data_2d_raw
    |   	|-- 2013_05_28_drive_0000_sync # train:[0, 2, 3, 4, 5, 7, 10] + val:[6] + test:[9]
    |   	|   |-- image_00
    |   	|   |   |-- data_rect # RGB images for left camera
    |   	|   |   |   |-- 000000.png
    |   	|   |   |   |-- 000001.png
    |   	|   |   |   |-- ...
    |   	|   |   |-- timestamps.txt
    |   	|   |-- image_01
    |   	|   |   |-- data_rect # RGB images for right camera
    |   	|   |   |   |-- 000000.png
    |   	|   |   |   |-- 000001.png
    |   	|   |   |   |-- ...
    |   	|   |   |-- timestamps.txt
    |   	|   |-- voxels # voxelized point clouds
    |   	|   |   |-- 000000.bin # voxelized input
    |   	|   |   |-- 000000.invalid # voxelized invalid mask
    |   	|   |   |-- 000000.label  #voxelized label
    |   	|   |   |-- 000005.bin # calculate every 5 frames 
    |   	|   |   |-- 000005.invalid
    |   	|   |   |-- 000005.label
    |   	|   |   |-- ...
    |   	|   |-- cam0_to_world.txt
    |   	|   |-- pose.txt # car pose information
    |   	|-- ...
    |   	|-- 2013_05_28_drive_0010_sync 
    |-- labels
    |       |-- 2013_05_28_drive_0000_sync 
    |       |   |-- 000000_1_1.npy # original labels
    |       |   |-- 000000_1_8.npy # 8x downsampled labels
    |       |   |-- 000005_1_1.npy
    |       |   |-- 000005_1_8.npy
    |       |   |-- ...
    |       |-- ... 
    |       |-- 2013_05_28_drive_0010_sync
    |-- labels_half # not unified, downsampled 
    |       |-- 2013_05_28_drive_0000_sync 
    |       |   |-- 000000_1_1.npy # original labels
    |       |   |-- 000000_1_8.npy # 8x downsampled labels
    |       |   |-- 000005_1_1.npy
    |       |   |-- 000005_1_8.npy
    |       |   |-- ...
    |       |-- ... 
    |       |-- 2013_05_28_drive_0010_sync
    |-- unified # unified
    |       |-- labels
    |           |-- 2013_05_28_drive_0000_sync 
    |           |   |-- 000000_1_1.npy # original labels
    |           |   |-- 000000_1_8.npy # 8x downsampled labels
    |           |   |-- 000005_1_1.npy
    |           |   |-- 000005_1_8.npy
    |           |   |-- ...
    |           |-- ... 
    |           |-- 2013_05_28_drive_0010_sync
    |-- calibration # preprocessed downsampled labels
    |   |-- calib_cam_to_pose.txt
    |   |-- calib_cam_to_velo.txt
    |   |-- calib_sick_to_velo.txt
    |   |-- image_02.yaml
    |   |-- image_03.yaml
    |   |-- perspective.txt
    |-- depth
     		|-- sequences
     			|-- 2013_05_28_drive_0000_sync
     			|	|-- 000000.npy
     			|	|-- 000001.npy
     			|-- ...
    			|-- 2013_05_28_drive_0010_sync
```

