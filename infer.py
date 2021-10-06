import torch
import numpy as np
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from tqdm import tqdm
import cv2

# from dataset import build_data_loader
# from models import build_model
from models import PAN
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter, Corrector


def test(test_loader, model, cfg):
    model.eval()

    for idx, data in enumerate(tqdm(test_loader)):
        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))

        # forward
        with torch.no_grad():
            outputs = model(**data)


        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)

from models import PAN

def main(args):
    cfg = Config.fromfile(args.config)

    # model
    param = dict()
    for key in cfg.model:
        if key == 'type':
            continue
        param[key] = cfg.model[key]
    model = PAN(**param)
    
    # model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(args.checkpoint)
            print(checkpoint.keys())

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
            
            # state = dict(state_dict=model.state_dict)
            # torch.save(model.state_dict(), 'checkpoints/pan_r18_alldata2.pth.tar')
            
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # test
    # test(test_loader, model, cfg)


if __name__ == '__main__':
    # print(1)
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', type=str, nargs='?',help='config file path', default='config/pan/pan_r18_ic15.py')
    parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_alldata2/checkpoint.pth.tar')
    # parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_alldata2.pth.tar')
    args = parser.parse_args()

    # print(args.checkpoint)
    # print(args.config)
    main(args)
    
    # cfg = Config.fromfile(args.config)
    # print(cfg['model'])
