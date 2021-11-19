## Recommended environment
```
Python 3.6+
Pytorch 1.1.0
torchvision 0.3
mmcv 0.2.12
editdistance
Polygon3
pyclipper
opencv-python 3.4.2.17
Cython
```

## Install
```shell script
pip install -r requirement.txt
./compile.sh
```
## Dataset
See [ICDAR2015 format](https://rrc.cvc.uab.es/?ch=4&com=tasks).
```none
pan_pp.pytorch
└── data
    ├── finetune
    │   ├── train
    │   │   ├── img
    │   │   └── gt
    │   └── test
    │       ├── img
    │       └── gt
```

## Training
```shell script

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```
