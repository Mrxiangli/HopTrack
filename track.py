import argparse, cv2
import os
import argparse
import os,sys
import time
from loguru import logger
from utils import EXP, Predictor

sys.path.insert(0, '/data/xiang/video_collab/para_det_trk')

import cv2
import torch
import math

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


################################################
#
# A trial run on the model to calculate the mainlin inference latency
# and use the mainline latency to determine \tau
#
################################################
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


################################################
#
# This functon takes a video source as input and output the 
# the frames at a cutomized rate or the video's original fps.
#
################################################
def frame_sampler(source, path, predictor, vis_folder, args):
    
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
    video_start = time.time()
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            print(outputs)

            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                print(f"online: {online_targets}")
                # if frame_id == 50:
                #     import sys
                #     sys.exit()
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_class_id = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_class_id.append(t.class_id)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                online_im = plot_tracking(
                    image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                )
            else:
                online_im = img_info['raw_img']


            # if (frame_idx % num_track) == 0:
            #     outputs, img_info = predictor.inference(frame)
            #     print(outputs)
            #     print(img_info)
            #     result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            # else:
            #     result_frame = frame
            
            frame_id +=1
           #frame_id = frame_id % num_track

            if args.save_result:
                vid_writer.write(online_im)
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
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

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