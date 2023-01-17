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

from yolox.data.datasets import COCO_CLASSES
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.tracker.byte_tracker import STrack, BYTETracker, joint_stracks

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    # explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    config = builder.create_builder_config()
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    config.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

        # generate TensorRT engine optimized for the target platform
        print('Building an engine...')
        print("num layers:", network.num_layers)
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        engine = builder.build_engine(network, config)
        context = engine.create_execution_context()
        print("Completed creating Engine")

        return engine, context


def main():
    ONNX_FILE_PATH = 'trt.onnx'

    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            # print(input_size)
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    img = '000032.jpg'
    test_size = (640, 640)
    preproc = ValTransform(legacy=False)

    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = os.path.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    img_info["ratio"] = ratio

    img, _ = preproc(img, None, test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    # img = img.float().cuda()
    # if self.device == "gpu":
    #     img = img.cuda()
    #     if self.fp16:
    #         img = img.half()  # to FP16

    # with torch.no_grad():
    #     t0 = time.time()
    #     outputs = self.model(img)
    #     print(outputs.shape)
    #     outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
    #     logger.info("Infer time: {:.4f}s".format(time.time() - t0))

    host_input = np.array(img.contiguous(), dtype=np.float32, order='C')

    # preprocess input data
    # host_input = np.array(torch.tensor(cv2.resize(cv2.imread('000032.jpg'), (640, 640), interpolation = cv2.INTER_AREA)).unsqueeze(0).contiguous(), dtype=np.float32, order='C')
    # print(host_input.shape)
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # print(device_output.shape)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    print(type(host_output))
    print(torch.Tensor(host_output).shape)
    print(host_output.shape)
    print(host_output)

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, 8400, 85)
    print(output_data.shape)
    outputs = postprocess(output_data, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)

    print(type(outputs[0]))
    print(outputs[0].shape)



if __name__ == '__main__':

    main()

    # parser = argparse.ArgumentParser()
    # # detection arg
    # parser.add_argument('-s', "--source", type=str, default="video", help="video or webcam")
    # parser.add_argument('-p', "--path", type=str, default=None, help="choose a video")
    # parser.add_argument("-m", "--model", type=str, default=None, help="model name")
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    # parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    # parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    # parser.add_argument("--tsize", default=None, type=int, help="test img size")
    # parser.add_argument("--save_result", action="store_true", help="whether to save the inference result")
    # parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
    #                     help="Adopting mix precision evaluating.")
    # parser.add_argument("--legacy", dest="legacy", default=False, action="store_true",
    #                     help="To be compatible with older versions")
    # # tracking arg
    # parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    # parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    # parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    # parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
    #                     help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    # parser.add_argument('--min_box_area', type=float, default=4, help='filter out tiny boxes')
    #
    # args = parser.parse_args()
    #
    # file_name = os.path.join(args.source, args.model)
    # vis_folder = None
    # if args.save_result:
    #     vis_folder = os.path.join(file_name, "vis_res")
    #     os.makedirs(vis_folder, exist_ok=True)
    #
    # # create a experiment object
    # yolox_dic = {
    #     "yolox_nano": (0.33, 0.25),
    #     "yolox_tiny": (0.33, 0.375),
    #     "yolox_s": (0.33, 0.5),
    #     "yolox_m": (0.67, 0.75),
    #     "yolox_l": (1.0, 1.0),
    #     "yolox_x": (1.33, 1.25)
    # }
    # if "yolox" in args.model:
    #     model_depth, model_width = yolox_dic[args.model]
    # exp = EXP(model_depth, model_width)
    #
    # vis_folder = None
    # if args.save_result:
    #     vis_folder = os.path.join(file_name, "vis_res")
    #     os.makedirs(vis_folder, exist_ok=True)
    #
    # logger.info("Args: {}".format(args))
    #
    # if args.conf is not None:
    #     exp.test_conf = args.conf
    # if args.nms is not None:
    #     exp.nmsthre = args.nms
    # if args.tsize is not None:
    #     exp.test_size = (args.tsize, args.tsize)
    #
    # model = exp.get_model()
    #
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #
    # if torch.cuda.is_available():
    #     device = "gpu"
    # else:
    #     device = "cpu"
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #     if args.fp16:
    #         model.half()  # to FP16
    # model.eval()
    #
    # logger.info("loading checkpoint")
    # ckpt = torch.load(args.ckpt, map_location="cpu")
    # # load the model state dict
    # model.load_state_dict(ckpt["model"])
    # logger.info("loaded checkpoint done.")
    #
    # ONNX_FILE_PATH = 'trt.onnx'
    #
    # x = torch.rand(1, 3, 640, 640).cuda()
    # torch.onnx.export(model, x, ONNX_FILE_PATH,
    #                   input_names=['input'], output_names=['output'], export_params=True, opset_version=11)
    #
