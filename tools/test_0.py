
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


# Define default arguments as global variables
DEFAULT_ARGS = {
    "demo": "image",
    "experiment_name": None,
    "name": None,
    "path": "./assets/dog.jpg",
    "camid": 0,
    "save_result": False,
    "exp_file": None,
    "ckpt": None,
    "device": "cpu",
    "conf": 0.3,
    "nms": 0.3,
    "tsize": None,
    "fp16": False,
    "legacy": False,
    "fuse": False,
    "trt": False,
}

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("demo", default=DEFAULT_ARGS["demo"], help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=DEFAULT_ARGS["experiment_name"])
    parser.add_argument("-n", "--name", type=str, default=DEFAULT_ARGS["name"], help="model name")
    parser.add_argument("--path", default=DEFAULT_ARGS["path"], help="path to images or video")
    parser.add_argument("--camid", type=int, default=DEFAULT_ARGS["camid"], help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true", default=DEFAULT_ARGS["save_result"], help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=DEFAULT_ARGS["exp_file"], type=str, help="please input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=DEFAULT_ARGS["ckpt"], type=str, help="ckpt for eval")
    parser.add_argument("--device", default=DEFAULT_ARGS["device"], type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=DEFAULT_ARGS["conf"], type=float, help="test conf")
    parser.add_argument("--nms", default=DEFAULT_ARGS["nms"], type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=DEFAULT_ARGS["tsize"], type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=DEFAULT_ARGS["fp16"], help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", action="store_true", default=DEFAULT_ARGS["legacy"], help="To be compatible with older versions")
    parser.add_argument("--fuse", dest="fuse", action="store_true", default=DEFAULT_ARGS["fuse"], help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", action="store_true", default=DEFAULT_ARGS["trt"], help="Using TensorRT model for testing.")
    return parser

# Simulate args using the global DEFAULT_ARGS
class Args:
    def __init__(self, defaults):
        for key, value in defaults.items():
            setattr(self, key.replace('-', '_'), value)








if __name__ == "__main__":
    args = Args(DEFAULT_ARGS)
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
