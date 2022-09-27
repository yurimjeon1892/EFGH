
# EFGH

This repository contains the code (in PyTorch) for "EFGHNet: A Versatile Image-to-Point Cloud Registration Network for Extreme Outdoor Environment" paper (IROS 2022).

## Requirements

* Python 3.8
* PyTorch 1.10
* RELLIS-3D dataset

## Environment

```
conda create -n efgh python=3.8
conda activate efgh
pip install -r requirements.txt
```

## Data
Download RELLIS-3D dataset from https://unmannedlab.github.io/research/RELLIS-3D
```
data
└── RELLIS-3D
    ├── RELLIS-3D
    |   ├── 00000
    |   |   ├── os1_cloud_node_kitti_bin
    |   |   ├── pylon_camera_node
    |   |   ├── calib.txt
    |   |   ├── poses.txt
    |   |   └── camera_info.txt    
    |   ├── 00001
    |   └── ..
    ├── RELLIS_3D
    |   ├── 00000
    |   |   └── transforms.yaml
    |   ├── 00001
    |   └── ..
    ├── pt_test.lst
    ├── pt_train.lst
    └── pt_val.lst
```

## Set up
```
cd lib 
python build_khash_cffi.py 
cd ..
```

## Train
Set data_root and ckpt_dir in the train_rellis.yaml file.
```
python main.py configs/train_rellis.yaml
```

## Test
Set ckpt_path in the test_rellis.yaml file.
```
python main.py configs/test_rellis.yaml
```


## Acknowledgements
Our BCL implementation is based on https://github.com/laoreja/HPLFlowNet. 
