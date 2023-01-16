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


################################################
#
# A trial run on the model to calculate the mainlin inference latency
# and use the mainline latency to determine \tau
#
################################################
def trail_run(predictor, frame, fps):
    for i in range(5):
        outputs, img_info = predictor.inference(frame)
    start = time.time()
    outputs, img_info = predictor.inference(frame)
    #result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
    end = time.time()
    num_track = math.ceil((end-start)/(1/fps))
    print("system preheat")
    return num_track


################################################
#
# This functon takes a video source as input and output the 
# the frames at a cutomized rate or the video's original fps.
#
################################################
def frame_sampler(source, path, predictor, vis_folder, args):

    ## Used for MOTA/MOTP etc.
    detection_results = defaultdict(dict)

    cls_names = COCO_CLASSES

    if source == "webcam":
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
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

    ret = False

    while ret == False:
        ret, frame = cap.read()

    num_track = trail_run(predictor, frame, fps)
    tracker = BYTETracker(args, frame_rate=math.ceil(fps))
    
    frame_id = 0
    results = []
    img_info = {}
    online_targets = []
    video_start = time.time()

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            print(f"=====================================================frame {frame_id}============================================")
            height, width = frame.shape[:2]
            img_info["height"] = height
            img_info["width"] = width
            img_info["raw_img"] = frame

            if frame_id % 10 == 0:
                light_track_id = []
                multiTracker = cv2.legacy.MultiTracker_create()

                #print(f"  ===================================================Inference========================================")
                outputs, img_info = predictor.inference(frame)
                #print(f"detection result:\n\t{outputs}")
                # the mean in the kalman filter object has all 8 states, potentially expose that to the following interface and do prediction.
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, frame_id)
                    #print(f"online_targets:\n\t{online_targets}")
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_class_id = []
                    for t in online_targets:
                        if t.tlwh[2] * t.tlwh[3] > args.min_box_area:
                            if sum(t.mean[4:]) == 0.0:
                                multiTracker.add(cv2.legacy.TrackerMedianFlow_create(), frame, t.tlwh)
                                light_track_id.append(t.track_id)
                            online_tlwhs.append(t.tlwh)
                            online_ids.append(t.track_id)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            detection_results[frame_id][t.track_id] = list(t.tlwh)
                    #print("==============plot track=================")
                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )
                    #online_im = img_info['raw_img']
                else:
                    online_im = img_info['raw_img']
                #print(f"========================================================================================================")

            else:
                #print(f" =====================================================Track============================================")
                # start = time.time()
                # print("Light track,", light_track_id)
                (success, bboxes) = multiTracker.update(frame)
                # end = time.time()
                # print('MedianFlow time, ', end - start)
                STrack.multi_predict(joint_stracks(online_targets, tracker.lost_stracks))
                #print(success, light_track_id)
                for track in online_targets:
                    track.frame_id = frame_id
                    if not success or track.track_id not in light_track_id:
                        pass
                    else:
                        light_id = light_track_id.index(track.track_id)
                        new_bbox = bboxes[light_id]
                        track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, track.tlwh_to_xyah(new_bbox))

                #print(f"online_targets:\n\t{online_targets}")

                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_class_id = []
                for t in online_targets:
                    if t.tlwh[2] * t.tlwh[3] > args.min_box_area:
                        online_tlwhs.append(t.tlwh)
                        online_ids.append(t.track_id)
                        online_scores.append(t.score)
                        online_class_id.append(t.class_id)
                        detection_results[frame_id][t.track_id] = list(t.tlwh)
                online_im = plot_tracking(
                    image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                )
                #online_im = img_info['raw_img']

            frame_id += 1

            # if frame_id == 900:
            #     break

            if args.save_result:
                vid_writer.write(online_im)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    video_end = time.time()
    print(f"video processing time is {video_end - video_start}")

    cap.release()
    cv2.destroyAllWindows()

    # out_file = open(f"{os.path.basename(args.path).split('.')[0]}.json", "w")
  
    # json.dump(detection_results, out_file, indent=6)
      
    # out_file.close()


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



