# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import cv2
import torch.nn.functional as F
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/root/pytorch-ppliteSeg/configs/liteseg_infer.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        # model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
        model_state_file = os.path.join(final_output_dir, 'best.pth')    
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load('/root/DDRNet.Pytorch/output2/face/pplite/best.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    newstate_dict = {k:v for k,v in pretrained_dict.items() if k in model.state_dict()}
  #  print(pretrained_dict.keys())

    model.load_state_dict(newstate_dict)
#    example_input = torch.randn(1, 3, 512, 512) # 根据你的模型输入进行调整
#    model.eval()
#    traced_model = torch.jit.trace(model, example_input)
#    traced_model.save("test.pt")    
    model = model.cuda()
    model.eval()


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    img = cv2.imread("1655.bmp")
    image1 = img.copy()
    stat1 = time.time()
    img = cv2.resize(img,(512,512))
    image = img.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std

    image = image.transpose((2,0,1))
    image = torch.from_numpy(image)
    
    

    image = image.unsqueeze(0)

    image = image.cuda()
    stat2 = time.time()
    out= model(image)
    out0 = out.squeeze(dim=0)
    out0 = F.softmax(out0,dim=0)


    out0 = torch.argmax(out0,dim=0)
  
    pred0 = out0.detach().cpu().numpy()
   
    pred0 = pred0*255
    pred_ch = np.zeros(pred0.shape)
    pred_rgb0 = np.array([pred_ch,pred_ch,pred0])

    pred_rgb0 = pred_rgb0.transpose(1,2,0)
    pred0 = cv2.resize(pred_rgb0,(image1.shape[1],image1.shape[0]))
    pred0 = pred0[:,:,-1]
    contours, hierarchy = cv2.findContours(pred0.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, contours, -1, (0, 0, 255), 2)

    cv2.imwrite("a222.jpg",image1)





if __name__ == '__main__':
    main()
