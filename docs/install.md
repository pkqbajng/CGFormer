# Step-by-step installation instructions

CGFormer is developed based on the official OccFormer codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate**

python 3.8 may not be supported.

```shell
conda create -n environment python=3.7 -y
conda activate environment
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

or 

```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

We select this pytorch version because mmdet3d 0.17.1 do not supports pytorch >= 1.11 and our cuda version is 11.3.

**c. Install mmcv, mmdet, and mmseg**

```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**c. Install mmdet3d 0.17.1 and DFA3D**

Compared with the offical version, the mmdetection3d provided by [OccFormer](https://github.com/zhangyp15/OccFormer) further includes operations like bev-pooling, voxel pooling. 

```shell
cd packages
bash setup.sh
cd ../
```

**d. Install other dependencies, like timm, einops, torchmetrics, spconv, pytorch-lightning, etc.**

```shell
pip install -r docs/requirements.txt
```

**e. Fix bugs (known now)**

```shell
pip install yapf==0.40.0
pip3 install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels
```
