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

        print(outputs.keys())
        img = data['imgs'].cpu().numpy()[0].transpose()
        img = ((1+img)*128).astype(int)
        # print(img)
        cv2.imwrite('test.jpg', img)
        print(outputs['bboxes'])
        if idx == 0: break
        
        if cfg.report_speed:
            report_speed(outputs, speed_meters)
        # post process of recognition
        if with_rec:
            outputs = pp.process(outputs)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)


def get_model(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    # print(json.dumps(cfg._cfg_dict, indent=4))
    # sys.stdout.flush()
    
    # model
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
    model.eval()
    return model, cfg

def get_input(img_path):
    img = cv2.imread(img_path)[:, :, [2,1,0]]
    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )

    # short_size = 736
    # img = scale_aligned_short(img, short_size)
    img_meta.update(dict(
        img_size=np.array(img.shape[:2])
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze_(0)

    data = dict(
        imgs=img,
        img_metas=img_meta
    )
    return data

def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms
    
    # print(1)
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', type=str, nargs='?', help='config file path', default='config/pan/pan_r18_ic15.py')
    parser.add_argument('checkpoint', nargs='?', type=str, default='/home/dev/Downloads/phi/pan_pp.pytorch/checkpoints/pan_r18_alldata/checkpoint.pth.tar')
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    model, cfg = get_model(args)
    
    src = '../example_text_detection'
    dic = 'result/res_pan_r18_alldata'
    # dic = 'result/res_pan_r18_mlt'
    # dic = 'result/res_pan_r18_ic15'
    img_list = os.listdir(src)
    
    for i in tqdm(range(len(img_list))):
        img_path = os.path.join(src, img_list[i])
        data_input = get_input(img_path)
        
        data_input['imgs'] = data_input['imgs'].cuda()
        data_input.update(dict(cfg=cfg))
        
        # print('original size', )
        # print('data size', data_input['imgs'].data.cpu().numpy().shape)
        # start = time.time()
        with torch.no_grad():
            outputs = model(**data_input)
        poly = outputs['bboxes']
        # print('num of word', len(poly))
        # print('time consumed', time.time()-start)
        
        
        
        img = cv2.imread(img_path)
        img_save = cv2.polylines(img, [box.reshape((4,2)) for box in poly], True, (0,255,0), 1)
        cv2.imwrite(os.path.join(dic, img_list[i]), img_save)
        # cv2.imwrite('test_shrink0.5.jpg', img_save)
    
    
    
    
