import argparse, cv2
import os
import argparse
import os,sys
import time
from loguru import logger
from utils import EXP, Predictor
import json
import numba
import threading

sys.path.insert(0, '/data/xiang/video_collab/para_det_trk')

import cv2
import torch
import math
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

@numba.jit()
def trajectory_finder(track):
    # get an hint of the moving direction
    tl_x, tl_y, width, height = track._tlwh
    vx = track.mean[4]
    vy = track.mean[5]
    vel_ratio = abs(vx)/abs(vy)
    
    # moving percentage as 3% 
    y_dis = 0.04 * height
    x_dis = y_dis * vel_ratio

    if abs(vx) > 0.5 and abs(vy) > 0.5:
        x_dir = 1 if vx > 0 else -1
        y_dir = 1 if vy > 0 else -1
    else:
        x_dir = 0
        y_dir = 0
    
    tl_x_new = int(tl_x + x_dir * x_dis)
    tl_y_new = int(tl_y + y_dir * y_dis)

    return int(tl_x_new), int(tl_y_new), int(width), int(height)

@numba.jit()
def pixel_distribution(img, tl_x, tl_y, width, height):
    new_img = img[tl_y:tl_y+height, tl_x:tl_x+width]
    hist = cv2.calcHist([new_img],[0],None,[256],[0,256])
    cv2.normalize(hist,hist,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def trail_run(predictor, frame, fps):
    for i in range(5):
        outputs, img_info = predictor.inference(frame)
    print("system preheat")
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
    print(f"current frame rate: {fps} fps")

    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(save_folder, exist_ok=True)

        if source == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")

        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    ret_val, frame = cap.read()
    num_track = trail_run(predictor, frame, fps)               #uncomment this when testing latency
    tracker = BYTETracker(args, frame_rate=math.ceil(fps))
    light_multi_tracker = cv2.MultiTracker_create()
    detection_result = {}
    
    frame_id = 0
    results = []
    video_start = time.time()
    img_info = {}
    img_info["height"] = height
    img_info["width"] = width
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # for mota measurement purpose
            if frame_id not in detection_result.keys():
                detection_result[frame_id]={}

           # print(f"=====================================================frame {frame_id}============================================")
            
            img_info["raw_img"] = frame

            if frame_id % 10 == 0:
                det_start = time.time()
                light_multi_tracker.clear()
                light_multi_tracker = cv2.MultiTracker_create()
                outputs, img_info = predictor.inference(frame)

                # the mean in the kalman filter object has all 8 states, potentially expose that to the following interface and do prediction.
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

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
                       # print(f"tid: {tid}  tlwh: {tlwh}")
                        tlbr = np.asarray(tlwh).copy()
                        tlbr[2:] += tlbr[:2]
                        tlbr=np.append(tlbr,[t.score, t.class_id])
                        tlbr=np.append(tlbr,[tid,t.mean])
                        predicted_bbox.append(tlbr)

                        if tlwh[2] * tlwh[3] > args.min_box_area: 
                            t.color_dist = pixel_distribution(frame, int(t.tlwh[0]), int(t.tlwh[1]), int(t.tlwh[2]), int(t.tlwh[3]))
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            
                            if t.first_seen:
                                light_tracker_list.append((int(t.tlwh[0]),int(t.tlwh[1]),int(t.tlwh[2]),int(t.tlwh[3])))
                                light_tracker_id.append(tid)
                            detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])

                #    print(f"light list: {light_tracker_list}")
                    for each in light_tracker_list:
                        light_multi_tracker.add(cv2.TrackerMedianFlow_create(), frame, each)

                 #   print("=======plot track=================")
                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )
                   # print(f"det time: {(time.time()-det_start)*1000}")

                else:
                    online_im = img_info['raw_img']


            else:
                # for mota purpose
                if frame_id not in detection_result.keys():
                    detection_result[frame_id]={}
                track_start = time.time()
                if len(predicted_bbox) != 0:
                    
                    predict_bbox = []
                    if frame_id % 10 == 3:
                        light_track_ok, light_track_bbox = light_multi_tracker.update(frame)
                    else:
                        light_track_ok = False

                    tk_start = time.time()
                    for idx, each_track in enumerate(online_targets):
                      #  print(f"track_id: {each_track.track_id}  mean: {each_track.mean}")
                        
                        # check if the bbox is in the light_tracker_list -> new track or reinitiated track 
                        if light_track_ok and each_track.first_seen:
                            light_id = light_tracker_id.index(each_track.track_id)
                            new_bbox = light_track_bbox[light_id]
                            each_track.first_seen = False
                            each_track.kalman_adjust = True
                            color_dist = pixel_distribution(frame, int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]))
                            each_track.dist_threshold = cv2.compareHist(each_track.color_dist, color_dist, cv2.HISTCMP_CORREL)
                            # need to convert from tlwh to tlbr
                            predict_bbox.append([new_bbox[0], new_bbox[1], new_bbox[2]+new_bbox[0], new_bbox[3]+new_bbox[1], predicted_bbox[idx][4], predicted_bbox[idx][5]])
                        
                        elif each_track.kalman_adjust and each_track.kalman_adjust_period != 0:
                            tmp_traject_bbox = []
                            tmp_threshold = []
                            tmp_color_dist = []
                            for count in range(5):
                                tl_x_new, tl_y_new, width, height = trajectory_finder(each_track)
                                color_dist = pixel_distribution(frame, tl_x_new, tl_y_new, width, height)
                                tmp_thresh = cv2.compareHist(each_track.color_dist, color_dist, cv2.HISTCMP_CORREL)
                                tmp_traject_bbox.append([tl_x_new, tl_y_new, width, height])
                                tmp_threshold.append(tmp_thresh)
                            tmp_index = tmp_threshold.index(max(tmp_threshold))
                            tmp_bbox = tmp_traject_bbox[tmp_index]
                            each_track.kalman_adjust_period -= 1
                            if each_track.kalman_adjust_period == 0:
                                each_track.kalman_adjust = False
                            predict_bbox.append([tmp_bbox[0], tmp_bbox[1], tmp_bbox[2]+tmp_bbox[0], tmp_bbox[3]+tmp_bbox[1], predicted_bbox[idx][4], predicted_bbox[idx][5]])
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
                 #   print(f"tk_time: {(time.time() - tk_start)*1000}")      

                    predicted_bbox = torch.tensor(np.array(predict_bbox), dtype=torch.float32)


                    # using the predicted bbox as the new detection result and feed into the tracker update
                    tk_update = time.time()
                    online_targets = tracker.new_update(predicted_bbox, [img_info['height'], img_info['width']], exp.test_size)
                 #   print(f"tk_update: {(time.time()-tk_update)*1000}")
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
                            detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])

                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )
                  #  print(f"track time: {(time.time()-track_start)*1000}")
                else:
                    online_im = img_info['raw_img']
            
            frame_id +=1

            if args.save_result:
                vid_writer.write(online_im)
                cv2.imwrite(f'/data/xiang/video_collab/para_det_trk/imgs/{frame_id}.png', online_im)
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
    print(f"video processing time is {video_end - video_start}")
    with open("traffic_2.json",'w') as fp:
        json.dump(detection_result,fp)  
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # detection arg
    parser.add_argument('-s',"--source", type=str, default="video", help="video or webcam")
    parser.add_argument('-p', "--path", type=str, default=None, help="choose a video")
    parser.add_argument("-m", "--model", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    # tracking arg
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=4, help='filter out tiny boxes')

    args = parser.parse_args()

    file_name = os.path.join(args.source, args.model)
    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    # create a experiment object
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
    exp = EXP(model_depth, model_width)

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
        exp.test_size = (args.tsize, args.tsize)
    
    model = exp.get_model()
    
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"

    if torch.cuda.is_available():
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    predictor = Predictor(model, exp, COCO_CLASSES, device, args.fp16, args.legacy)

    frame_sampler(args.source, args.path, predictor, vis_folder, args)