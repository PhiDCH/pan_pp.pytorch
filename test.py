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

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter, Corrector


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    model.eval()

    with_rec = hasattr(cfg.model, 'recognition_head')
    if with_rec:
        pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_pa_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    for idx, data in enumerate(tqdm(test_loader)):
        # print('Testing %d/%d' % (idx, len(test_loader)))
        # sys.stdout.flush()

        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))

        # forward
        with torch.no_grad():
            outputs = model(**data)

        # print(data['imgs'].data.cpu().numpy().shape)
        # img = data['imgs'].data.cpu().numpy()[0].transpose()
        # img = ((1+img)*128).astype(int)
        # # print(img)
        # cv2.imwrite('test.jpg', img)
        # # print(outputs['bboxes'])
        # if idx == 0: break
        
        if cfg.report_speed:
            report_speed(outputs, speed_meters)
        # post process of recognition
        if with_rec:
            outputs = pp.process(outputs)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    # print(json.dumps(cfg._cfg_dict, indent=4))
    # sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(dict(
            voc=data_loader.voc,
            char2id=data_loader.char2id,
            id2char=data_loader.id2char,
        ))
    model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(test_loader, model, cfg)


if __name__ == '__main__':
    # print(1)
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', type=str, nargs='?',help='config file path', default='config/pan/pan_r18_ic15.py')
    parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_ic15/checkpoint.pth.tar')
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
