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
See [dataset](https://github.com/whai362/pan_pp.pytorch/tree/master/dataset).

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar
```
