from sklearn.cluster import DBSCAN
import argparse, cv2
import os
import argparse
import os,sys
import time
from loguru import logger
from utils import EXP, Predictor, build_engine
import json
import threading
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings


sys.path.insert(0, '/data/Video_Colab')

import cv2
import torch
import math
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer



def dbscan_clustering(online_targets):
    
    centroid_list = []
    tid_list = []
    for each_track in online_targets:
        tid, tlwh = each_track.track_id, each_track.tlwh
        x,y,w,h = tlwh
        tid_list.append(tid)
        centroid_list.append([x+w/2, y+h/2])
    cluster = DBSCAN(eps=50, min_samples=2).fit(centroid_list)
   # print(cluster.labels_)
    
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
  #  print(cluster_dic)
    return cluster_dic, cluster_num

def detection_rate_adjuster(cluster_dic, cluster_num):
    if cluster_num > 0 :
        detection_rate = 5
    else:
        detection_rate = 10
    return detection_rate
    

def trail_run(predictor, frame, fps):
    for i in range(5):
        outputs, img_info = predictor.inference(frame)
  #  print("system preheat")
    start = time.time()
    outputs, img_info = predictor.inference(frame)
    result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
    end = time.time()
    num_track = math.ceil((end-start)/(1/fps))
    return num_track


def frame_sampler(source, path, predictor, vis_folder, args):
    
    cls_names = COCO_CLASSES

    if source == "webcam":
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(save_folder, exist_ok=True)

        if source == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")

        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
        

   # ret_val, frame = cap.read()
   # num_track = trail_run(predictor, frame, fps)               #uncomment this when testing latency
    tracker = BYTETracker(args, frame_rate=math.ceil(fps))
    light_multi_tracker = cv2.MultiTracker_create()
    detection_result = {}
    
    frame_id = 1
    results = []
    video_start = time.time()
    img_info = {}
    img_info["height"] = height
    img_info["width"] = width
    detection_rate = 10
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # for mota measurement purpose
            if frame_id not in detection_result.keys():
                detection_result[frame_id]={}

           # print(f"=====================================================frame {frame_id}============================================")
            
            img_info["raw_img"] = frame

            if frame_id % detection_rate == 1:
                det_start = time.time()
                light_multi_tracker.clear()
                light_multi_tracker = cv2.MultiTracker_create()
                outputs, img_info = predictor.inference(frame)

                # the mean in the kalman filter object has all 8 states, potentially expose that to the following interface and do prediction.
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width'], img_info["raw_img"]], exp.test_size)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_class_id = []
                    predicted_bbox = []
                    light_tracker_list = []
                    light_tracker_id = []

                    for idx_t, t in enumerate(online_targets):
                        tlwh = t.tlwh
                        tid = t.track_id
                        tlbr = np.asarray(tlwh).copy()
                        tlbr[2:] += tlbr[:2]
                        tlbr=np.append(tlbr,[t.score, t.class_id])
                        tlbr=np.append(tlbr,[tid,t.mean])
                        predicted_bbox.append(tlbr)

                        if tlwh[2] * tlwh[3] > args.min_box_area: 
                           # t.color_dist = pixel_distribution(frame, int(t.tlwh[0]), int(t.tlwh[1]), int(t.tlwh[2]), int(t.tlwh[3]))
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            
                            if t.mean[4] == 0 and t.mean[5] == 0 and t.mean[6] == 0 and t.mean[7] == 0 :
                                light_tracker_list.append((int(t.tlwh[0]),int(t.tlwh[1]),int(t.tlwh[2]),int(t.tlwh[3])))
                                light_tracker_id.append(tid)
                            if t.class_id == 0:     # person only
                                
                                detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])
                            print(f"{frame_id},{t.track_id},{t.tlwh[0]},{t.tlwh[1]},{t.tlwh[2]},{t.tlwh[3]},{t.score},-1,-1,-1")

                    for each in light_tracker_list:
                        light_multi_tracker.add(cv2.TrackerMedianFlow_create(), frame, each)
                    
                    cluster_dic, cluster_num = dbscan_clustering(online_targets)
                    detection_rate = detection_rate_adjuster(cluster_dic, cluster_num)

                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )

                else:
                    online_im = img_info['raw_img']

            else:

                track_start = time.time()
                # for mota purpose
                if frame_id not in detection_result.keys():
                    detection_result[frame_id]={}
                track_start = time.time()
                if len(predicted_bbox) != 0:
                    
                    predict_bbox = []
                    if frame_id <= 1: # update three frames with light tracker to get the kalman filter upd
                        light_track_ok, light_track_bbox = light_multi_tracker.update(frame)
                    else:
                        light_track_ok = False

                    tk_start = time.time()
                    for idx, each_track in enumerate(online_targets):
                        
                        # check if the bbox is in the light_tracker_list -> new track or reinitiated track 
                        if light_track_ok and each_track.track_id in light_tracker_id:
                            light_id = light_tracker_id.index(each_track.track_id)
                            new_bbox = light_track_bbox[light_id]
                            predict_bbox.append([new_bbox[0], new_bbox[1], new_bbox[2]+new_bbox[0], new_bbox[3]+new_bbox[1], predicted_bbox[idx][4], predicted_bbox[idx][5]])
                        else:
                            new_x = each_track.mean[0]+each_track.mean[4]
                            new_y = each_track.mean[1]+each_track.mean[5]
                            new_a = each_track.mean[2]+each_track.mean[6]
                            new_h = each_track.mean[3]+each_track.mean[7]
                            new_w = new_a * new_h
                            tlwh = [new_x - new_w/2, new_y - new_h/2, new_w, new_h]

                            # converting the predicted tlwh to tlbr and use this as the new detection bbox
                            tlbr = np.asarray(tlwh).copy()
                            tlbr[2:] += tlbr[:2]
                            tlbr=np.append(tlbr,[predicted_bbox[idx][4], predicted_bbox[idx][5]])
                            predict_bbox.append(tlbr)   

                    predicted_bbox = torch.tensor(np.array(predict_bbox), dtype=torch.float32)


                    # using the predicted bbox as the new detection result and feed into the tracker update
                    tk_update = time.time()
                    online_targets = tracker.new_update(predicted_bbox, [img_info['height'], img_info['width'], img_info['raw_img']], exp.test_size)
                    # the following dynamic sampling might be enabled 
                    #cluster_dic, cluster_num = dbscan_clustering(online_targets)
                    #detection_rate = detection_rate_adjuster(cluster_dic, cluster_num)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_class_id = []
                    predicted_bbox = []

                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        tlbr = np.asarray(tlwh).copy()
                        tlbr[2:] += tlbr[:2]
                        tlbr=np.append(tlbr,[t.score, t.class_id])
                        tlbr=np.append(tlbr,[tid,t.mean])
                        predicted_bbox.append(tlbr)

                        if tlwh[2] * tlwh[3] > args.min_box_area: #and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            if t.class_id == 0:     # person only
                                detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])
                        print(f"{frame_id},{t.track_id},{t.tlwh[0]},{t.tlwh[1]},{t.tlwh[2]},{t.tlwh[3]},{t.score},-1,-1,-1")        

                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )
                    #print(f"track time: {(time.time()-track_start)*1000}")
                else:
                    online_im = img_info['raw_img']
            
            frame_id +=1

            if args.save_result:
                vid_writer.write(online_im)
               # cv2.imwrite(f'/home/dcsl/Documents/Video_Colab/imgs/{frame_id}.png', online_im)
                pass
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    video_end = time.time()
   # print(f"video processing time is {video_end - video_start}")
    with open(f"{args.path.split('.')[0]}_result.json",'w') as fp:
        json.dump(detection_result,fp)  
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # detection arg
    parser.add_argument('-s',"--source", type=str, default="video", help="video or webcam")
    parser.add_argument('-trt',"--trt", type=str, default=None, help="video or webcam")
    parser.add_argument('-p', "--path", type=str, default=None, help="choose a video")
    parser.add_argument("-m", "--model", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    parser.add_argument("--fuse",dest="fuse",default=False,action="store_true",help="Fuse conv and bn for testing.")
    parser.add_argument("--mot",dest="mot",default=False,action="store_true",help="Fuse conv and bn for testing.")
    # tracking arg
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=4, help='filter out tiny boxes')

    args = parser.parse_args()

    if args.trt != None:
        file_name = os.path.join(args.source, args.trt.split('.')[0])
    else:
        file_name = os.path.join(args.source, args.model)
    
    vis_folder = None

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    # create a experiment object
    if args.model.split("_")[0] == "yolox":
        yolox_dic={
            "yolox_nano":(0.33, 0.25),
            "yolox_tiny":(0.33, 0.375),
            "yolox_s":(0.33, 0.5),
            "yolox_m":(0.67, 0.75),
            "yolox_l":(1.0, 1.0),
            "yolox_x":(1.33, 1.25)
        }
        if "yolox" in args.model:
            model_depth, model_width = yolox_dic[args.model]
            if args.trt == None:
                exp = EXP(model_depth, model_width, args)
            else:
                exp = EXP(None, None, args)

    if args.model == "yolov5":
        model_depth, model_width = None, None
        exp = EXP(model_depth, model_width, args)
    
    if args.model == "mobilenet":
        pass

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = args.tsize

    if args.trt != None:
        model = None
        engine, context = build_engine(args.trt)

        for binding in engine:
            if engine.binding_is_input(binding):  # we expect only one input
                input_shape = engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                device_input = cuda.mem_alloc(input_size)
            else:  # and one output
                output_shape = engine.get_binding_shape(binding)
                # create page-locked memory buffers (i.e. won't be swapped to disk)
                host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
                device_output = cuda.mem_alloc(host_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        
        trt_engine = (device_input, device_output, host_output, stream, context, engine)

    else:
        model = exp.get_model()
        model.eval()

        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        # if torch.cuda.is_available():
        #     model.cuda()
        #     if args.fp16:
        #         model.half()  # to FP16

        if "yolox" in args.model:
            logger.info("loading checkpoint")
            ckpt = torch.load(args.ckpt, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")
            if args.fuse:
                model = fuse_model(model)
        trt_engine = (None, None, None, None, None, None)
    
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"


    predictor = Predictor(args, model, exp, trt_engine, COCO_CLASSES, device, args.fp16, args.legacy)

    frame_sampler(args.source, args.path, predictor, vis_folder, args)
