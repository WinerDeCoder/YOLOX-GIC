#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import os

# Append the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




import argparse
import os
import time
from loguru import logger

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def resize_image(input_path, output_path):
    # Đọc ảnh
    image = cv2.imread(input_path)
    
    width = 1080
    height = 651

    # Resize về kích thước mục tiêu
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Lưu ảnh đã resize
    cv2.imwrite(output_path, resized)

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        #print("4,2")
        img = str(img)
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            #print("4,3")
        else:
            img_info["file_name"] = None
            #print("4,4")

        print("img: ", img.shape)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        #print("4,5")

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        
        #print("4,6")

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        
        #print("4,7")
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
            )
        return outputs, img_info


def main(image_path):
    # Load the experiment and model configuration
    exp_file = None
    model_name = "yolox-s"
    conf_thresh = 0.25
    nms_thresh = 0.45
    tsize = 640
    device = "cpu"  # cpu
    ckpt_path = "app/detect_model/YOLOX/yolox_s.pth"
    
    print("1")

    exp = get_exp(exp_file, model_name)

    exp.test_conf = conf_thresh
    exp.nmsthre = nms_thresh
    exp.test_size = (tsize, tsize)
    
    print("2")

    model = exp.get_model()

    if device == "gpu":
        model.cuda()
    model.eval()
    
    print("3")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None, device, False, False
    )
    
    print("4")
    
    resize_image(image_path, image_path)
    
    print("4,1", image_path, type(image_path))

    outputs, img_info = predictor.inference(image_path)
    
    print("5")

    # Extract bounding boxes for class "person"
    person_bboxes = []
    if outputs[0] is not None:
        for output in outputs[0]:
            x1, y1, x2, y2, score, class_score, cls_id = output.cpu().numpy()
            if int(cls_id) == COCO_CLASSES.index("person"):
                person_bboxes.append([x1, y1, x2, y2, score])

    return person_bboxes

# Example usage from another script
if __name__ == "__main__":
    result = main("assets/CCCD_smaller.jpg")
    print("Detected bounding boxes:", result)