from sklearn.cluster import DBSCAN
import os
import random
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import time
from loguru import logger

from torch.nn import Module

from yolox.data.data_augment import ValTransform, preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

import sys
# for yolov5 detector support
yolov5_dir = "/".join(os.getcwd().split('/')[:-1])+"/yolov5"
sys.path.insert(0, yolov5_dir)

from models.common import DetectMultiBackend
from utils_yolov5.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils_yolov5.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils_yolov5.plots import Annotator, colors, save_one_box
from utils_yolov5.torch_utils import select_device, smart_inference_mode
from utils_yolov5.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste, letterbox, mixup, random_perspective)


class EXP:
    def __init__(self,depth,width, args):
        self.args = args

        # ---------------- model config ---------------- #
        if self.args.mot:                                # this is for MOT test only, pretrained model only focus on human
            self.num_classes = 1
            self.test_size = (800, 1440)
        else:
            self.num_classes = 80                        # this is for generic yolo series trained on COCO 
            self.test_size = (640, 640)

        self.depth = depth                              
        self.width = width
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65
        self.args = args
        self.device ="cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self):
        # build yolox architecture
        if "yolox" in self.args.model:
            def init_yolo(M):
                for m in M.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eps = 1e-3
                        m.momentum = 0.03

            if getattr(self, "model", None) is None:
                in_channels = [256, 512, 1024]
                backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
                head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
                self.model = YOLOX(backbone, head)

            self.model.apply(init_yolo)
            self.model.head.initialize_biases(1e-2)
            self.model.to(self.device)
            return self.model
        
        # build yolo5 architecture
        if "yolov5" in self.args.model:
            self.model = DetectMultiBackend(self.args.ckpt, device=torch.device(self.device), dnn=False, data='data/coco128.yaml', fp16=False)
            return self.model


class Predictor(object):
    def __init__(self, args, model, exp, trt_engine, cls_names=COCO_CLASSES, device="cpu", fp16=False, legacy=False):
        self.args = args
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.non_mot_preproc = ValTransform(legacy=legacy)
        self.device_input = trt_engine[0] 
        self.device_output = trt_engine[1] 
        self.host_output = trt_engine[2] 
        self.stream = trt_engine[3] 
        self.context = trt_engine[4] 
        self.engine = trt_engine[5]

        # used for input preprocee for yolox trained on MOT dataset 
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
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
    
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        
        # preprocess for input image of yolox trained on MOT dataset
        if "yolox" in self.args.model and self.args.mot:
            img, _ = preproc(img, self.test_size, self.rgb_means, self.std)
            img = torch.from_numpy(img).unsqueeze(0)

        # preprocess for default yolox model 
        if "yolox" in self.args.model and not self.args.mot:
            img, _ = self.non_mot_preproc(img, None, self.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
        
        # preprocess for default yolov5
        if "yolov5" in self.args.model:
            im = letterbox(img, 640, 32, True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
            im = im.half() if self.fp16 else im.float()
            img = im/255
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

        if self.args.trt != None:
            img = np.array(img.numpy(), dtype=np.float32, order='C')
        else:
            img = img.float()
            if self.device == "gpu":
                img = img.cuda()
                if self.fp16:
                    img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            # inference with tensorrt engine
            if self.args.trt != None:
                cuda.memcpy_htod_async(self.device_input, img, self.stream)
                self.context.execute_async(bindings=[int(self.device_input), int(self.device_output)], stream_handle=self.stream.handle)
                cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
                self.stream.synchronize()
                outputs = torch.Tensor(self.host_output).reshape(self.engine.max_batch_size, 8400, 85)
            else:
                outputs = self.model(img)

            if "yolox" in self.args.model:
                outputs = postprocess(outputs, self.num_classes, self.confthre,self.nmsthre, class_agnostic=True)
            if "yolov5" in self.args.model:
                outputs = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

                for det in outputs:
                    if len(det):
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], (1080,1920,3)).round()

            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

# build tensorrt engine if available
def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger()
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # only one frame in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        print("FP 16 mode")
        builder.fp16_mode = True
    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine, context

# Auto clustering for all available objects in the frame
def dbscan_clustering(online_targets):

    centroid_list = []
    tid_list = []
    for each_track in online_targets:
        tid, tlwh = each_track.track_id, each_track.tlwh
        x,y,w,h = tlwh
        tid_list.append(tid)
        centroid_list.append([x+w/2, y+h/2])
    cluster = DBSCAN(eps=50, min_samples=2).fit(centroid_list)
    
    cluster_dic = {}
    cluster_num = 0
    for idx, each in enumerate(cluster.labels_):
        if each != -1 and each not in cluster_dic.keys():
            cluster_dic[each]=[tid_list[idx]]
            cluster_num += 1
        else:
            if each != -1:
                cluster_dic[each].append(tid_list[idx])
            else:
                cluster_num += 1
    return cluster_dic, cluster_num

# Dynamic_rate adjuster to accomodate the process quality 
def detection_rate_adjuster(cluster_dic, cluster_num):
    if cluster_num > 0 :
        detection_rate = 7
    else:
        detection_rate = 8
    return detection_rate

# enable trail run when needed for GPU preheat!
def trail_run(predictor, frame, fps):
    for i in range(5):
        outputs, img_info = predictor.inference(frame)
    start = time.time()
    outputs, img_info = predictor.inference(frame)
    result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
    end = time.time()
    num_track = math.ceil((end-start)/(1/fps))
    return num_track