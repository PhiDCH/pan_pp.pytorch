## Install
```shell script
pip install -r requirement.txt
./compile.sh
```
## Dataset
See [ICDAR2015 format](https://rrc.cvc.uab.es/?ch=4&com=tasks).
```none
pan_pp.pytorch
└── icdar2015
    │   train
    │   ├── img
    │   └── gt
    │   test
    │   ├── img
    │   └── gt
```

## Download pretrain model
```
gdown --id 1dXMKs1VAltre6RCtDHYft-2q8oCtC_Yh -O checkpoint.pth.tar
```

## Training
```shell script

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```

See [Colab](https://colab.research.google.com/drive/153gGIa9zAKQskd6jOitc8emS3mfO6Ab_?usp=sharing).
