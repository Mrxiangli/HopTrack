
import argparse, cv2
import os
import argparse
import os,sys
import time
from loguru import logger
from utils import EXP, Predictor, build_engine, dbscan_clustering, detection_rate_adjuster, trail_run
import json
import threading
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings
import multiprocessing
from multiprocessing import Pool, Process

# add current folder into path
sys.path.insert(0, "/".join(os.getcwd().split('/')[:-1]))

import cv2
import torch
import math
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.hop_tracker import HOPTracker, pixel_distribution
from yolox.tracking_utils.timer import Timer

def dynamic_rate_adjuster(object_pool, rate_pool, term_pool, mot_test_file, sampling_strategy):
    prev = 0
    while True:
        if term_pool.qsize() != 0:
            break
        if object_pool.qsize() != 0:
            online_targets = object_pool.get() 
            if len(online_targets) !=0 :
                cluster_dic, cluster_num = dbscan_clustering(online_targets)
                detection_rate = detection_rate_adjuster(cluster_dic, cluster_num, mot_test_file, sampling_strategy)
                if detection_rate != prev:
                    rate_pool.put(detection_rate)
                    prev = detection_rate

def start_process(source, path, predictor, vis_folder, args):

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    # retrieve frame width and height
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    # save results 
    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(save_folder, exist_ok=True)

        if source == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")

        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
      
    tracker = HOPTracker(args, frame_rate=math.ceil(fps))
    light_multi_tracker = cv2.MultiTracker_create()
    detection_result = {}
    
    frame_id = 1
    results = []
    video_start = time.time()
    img_info = {}
    img_info["height"] = height
    img_info["width"] = width 

    if args.upper:
        sampling_strategy = 1           # upper
    elif args.lower: 
        sampling_strategy = 2           # lower
    else:
        sampling_strategy = 0           # dynamic

    detection_rate = 9
    if args.source != "webcam":
        mot_test_file = args.path.split('/')[-1]
        if "MOT16-05" in args.path.split('/')[-1] or "MOT16-06" in args.path.split('/')[-1]:
            detection_rate = 4
        if "MOT16-13" in args.path.split('/')[-1] or "MOT16-14" in args.path.split('/')[-1]:
            detection_rate = 7
        
    else:
        detection_rate = 9

    total_post_fuse = 0
    total_post_update = 0
    det_frame_ct = 0
    tk_frame_ct = 0
    object_pool = multiprocessing.Queue()
    rate_pool = multiprocessing.Queue()
    term_pool = multiprocessing.Queue()

    p=Process(target=dynamic_rate_adjuster, args=(object_pool,rate_pool,term_pool,mot_test_file, sampling_strategy))
    p.daemon = True
    p.start()
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # for mota measurement purpose
            if frame_id not in detection_result.keys():
                detection_result[frame_id]={}
            if args.source == "webcam":
                frame = cv2.resize(frame,(640,480))
            img_info["raw_img"] = frame

            if frame_id % detection_rate == 1:
                light_multi_tracker.clear()
                light_multi_tracker = cv2.MultiTracker_create()
                outputs, img_info = predictor.inference(frame)

                if outputs[0] is not None:
                    online_targets = tracker.detect_track_fuse(outputs[0], [img_info['height'], img_info['width'], img_info["raw_img"]], exp.test_size)

                    inter_process_start = time.time()
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_class_id = []
                    predicted_bbox = []
                    light_tracker_list = []
                    light_tracker_id = []

                    for idx_t, t in enumerate(online_targets):
                        t.last_detected_frame = frame_id
                        tlwh = t.tlwh
                        tid = t.track_id
                        tlbr = np.asarray(tlwh).copy()
                        tlbr[2:] += tlbr[:2]
                        tlbr=np.append(tlbr,[t.score, t.class_id])
                        tlbr=np.append(tlbr,[tid,t.mean])
                        predicted_bbox.append(tlbr)

                        if tlwh[2] * tlwh[3] > args.min_box_area and tlwh[3]/tlwh[2] >= args.aspect_ratio_thresh: 
                            hist_b, hist_g, hist_r = pixel_distribution(frame, int(t.tlwh[0]), int(t.tlwh[1]), int(t.tlwh[2]), int(t.tlwh[3]))
                            t.color_dist = [hist_b, hist_g, hist_r]
                            if t.mean[4] == 0 and t.mean[5] == 0 and t.mean[6] == 0 and t.mean[7] == 0 :
                                    light_multi_tracker.add(cv2.TrackerMedianFlow_create(), frame, (int(t.tlwh[0]),int(t.tlwh[1]),int(t.tlwh[2]),int(t.tlwh[3])))
                                    light_tracker_id.append(tid)

                    # discard first three detection fuse time due to numba initialization
                    end_process = (time.time()-inter_process_start)*1000
                    det_frame_ct+=1
                    if det_frame_ct > 3:
                        total_post_fuse += end_process
                       # logger.info("average post fuse time: {:.4f}ms".format(total_post_fuse/(det_frame_ct-3)))
                        if args.source != "webcam":
                            tracker.worksheet.write(frame_id, 1, end_process)
                    
                    object_pool.put(online_targets)
                    if rate_pool.qsize()!=0:
                        detection_rate = int(rate_pool.get())
                    # the follow logic are used for writing results or draw bounding boxes; will not be caluclated towards the processing time    
                    for idx_t, t in enumerate(online_targets):
                        tlwh = t.tlwh
                        tid = t.track_id
                        if tlwh[2] * tlwh[3] > args.min_box_area and tlwh[3]/tlwh[2] >= args.aspect_ratio_thresh: 
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            
                            if t.class_id == 0:     # person only
                                detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])
                            print(f"{frame_id},{t.track_id},{t.tlwh[0]},{t.tlwh[1]},{t.tlwh[2]},{t.tlwh[3]},{t.score},-1,-1,-1")

                   
                   
                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = COCO_CLASSES
                    )

                else:
                    online_im = img_info['raw_img']

            else:
                # for mota purpose
                if frame_id not in detection_result.keys():
                    detection_result[frame_id]={}
                
                if len(predicted_bbox) != 0:
                    tk_start = time.time()
                    predict_bbox = []

                    if frame_id % detection_rate <= 3: # update three frames with light tracker to get the kalman filter upd
                        light_track_ok, light_track_bbox = light_multi_tracker.update(frame)
                    else:
                        light_track_ok = False

                    for idx, each_track in enumerate(online_targets):
                        
                        # check if the bbox is in the light_tracker_list -> new track or reinitiated track 
                        if light_track_ok and each_track.track_id in light_tracker_id and len(light_track_bbox)!=0:
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
                    tk_tmp = time.time()
                    online_targets = tracker.hopping_update(predicted_bbox, [img_info['height'], img_info['width'], img_info['raw_img']], exp.test_size)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_class_id = []
                    predicted_bbox = []

                    tk_frame_ct += 1
                    tk_dur = (tk_tmp - tk_start)*1000
                    if tk_frame_ct > 1:
                        total_post_update += tk_dur
                        #logger.info("average post update time: {:.4f}ms".format(total_post_update/(tk_frame_ct-1)))
                        if args.source != "webcam":
                            tracker.worksheet.write(frame_id, 3, tk_dur)

                    # the follow logic are used for writing results or draw bounding boxes; will not be caluclated towards the processing time
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        tlbr = np.asarray(tlwh).copy()
                        tlbr[2:] += tlbr[:2]
                        tlbr=np.append(tlbr,[t.score, t.class_id])
                        tlbr=np.append(tlbr,[tid,t.mean])
                        predicted_bbox.append(tlbr)

                        if tlwh[2] * tlwh[3] > args.min_box_area and tlwh[3]/tlwh[2] >= args.aspect_ratio_thresh:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            if t.class_id == 0:     # person only
                                detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])
                        print(f"{frame_id},{t.track_id},{t.tlwh[0]},{t.tlwh[1]},{t.tlwh[2]},{t.tlwh[3]},{t.score},-1,-1,-1")        

                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = COCO_CLASSES
                    )
                else:
                    online_im = img_info['raw_img']
            
            frame_id +=1

            if args.save_result:
                #vid_writer.write(online_im)
               # cv2.imwrite(os.getcwd()+"/imgs/"+str(frame_id-1)+".png", online_im)
                pass
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    video_end = time.time()
    if args.source != "webcam":
        tracker.workbook.close()
        term_pool.put('T')
        p.join()

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
    parser.add_argument("--conf", default=0.4, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    parser.add_argument("--fuse",dest="fuse",default=False,action="store_true",help="Fuse conv and bn for testing.")
    parser.add_argument("--mot",dest="mot",default=False,action="store_true",help="run mot trained model.")
    parser.add_argument("--dis_traj",dest="dis_traj",default=False,action="store_true",help="Disable trajectory finding and dynamic matching")
    parser.add_argument("--dynamic",dest="dynamic",default=False,action="store_true",help="Enable content-aware dynamic sampling")
    parser.add_argument("--upper",dest="upper",default=False,action="store_true",help="Enable upper bound sampling, fast but lower mota")
    parser.add_argument("--lower",dest="lower",default=False,action="store_true",help="Enable lower bound sampling, slow but higher mota")


    # tracking arg
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
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
    start_process(args.source, args.path, predictor, vis_folder, args)
