import os
import sys
import cv2
import time
import math
import json
import argparse
import numpy as np
from loguru import logger
from utils import EXP, Predictor
from collections import defaultdict

import torch

from yolox.utils import get_model_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # detection arg
    parser.add_argument("-m", "--model", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")

    args = parser.parse_args()

    # create a experiment object
    yolox_dic = {
        "yolox_nano": (0.33, 0.25),
        "yolox_tiny": (0.33, 0.375),
        "yolox_s": (0.33, 0.5),
        "yolox_m": (0.67, 0.75),
        "yolox_l": (1.0, 1.0),
        "yolox_x": (1.33, 1.25)
    }
    if "yolox" in args.model:
        model_depth, model_width = yolox_dic[args.model]
    exp = EXP(model_depth, model_width)

    logger.info("Args: {}".format(args))

    model = exp.get_model().cuda()
    model.eval()

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    ONNX_FILE_PATH = args.model + '.onnx'

    x = torch.rand(1, 3, 640, 640).cuda()
    torch.onnx.export(model, x, ONNX_FILE_PATH,
                      input_names=['input'], output_names=['output'], export_params=True, opset_version=11)
