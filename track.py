import argparse, cv2
import os
import argparse
import os,sys
import time
from loguru import logger
from utils import EXP, Predictor
import json

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

    #num_track = trail_run(predictor, frame, fps)               #uncomment this when testing latency
    tracker = BYTETracker(args, frame_rate=math.ceil(fps))
    light_multi_tracker = cv2.MultiTracker_create()
    detection_result = {}
    
    frame_id = 0
    results = []
    video_start = time.time()
    img_info ={}
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # for mota purpose
            if frame_id not in detection_result.keys():
                detection_result[frame_id]={}

            print(f"=====================================================frame {frame_id}============================================")
            height, width = frame.shape[:2]
            img_info["height"] = height
            img_info["width"] = width
            img_info["raw_img"] = frame
            if frame_id % 10 == 0:
                light_multi_tracker.clear()
                light_multi_tracker = cv2.MultiTracker_create()
                outputs, img_info = predictor.inference(frame)
                print(f"detection result: {outputs}")
                # the mean in the kalman filter object has all 8 states, potentially expose that to the following interface and do prediction.
                if outputs[0] is not None:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                    print(f"online_targets: {online_targets}")
                    # if frame_id == 50:
                    #     import sys
                    #     sys.exit()
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
                        print(f"tid: {tid}  tlwh: {tlwh}")
                        tlbr = np.asarray(tlwh).copy()
                        tlbr[2:] += tlbr[:2]
                        tlbr=np.append(tlbr,[t.score, t.class_id])
                        tlbr=np.append(tlbr,[tid,t.mean])
                        predicted_bbox.append(tlbr)

                        if tlwh[2] * tlwh[3] > args.min_box_area: 
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            if t.mean[4] < 0.00001 and  t.mean[5] < 0.00001 and t.mean[6] < 0.00001 and t.mean[7] < 0.00001:
                                light_tracker_list.append((t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3]))
                                light_tracker_id.append(tid)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                            detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])

                    print(f"light list: {light_tracker_list}")
                    for each in light_tracker_list:
                        light_multi_tracker.add(cv2.TrackerMedianFlow_create(), frame, each)

                    print("=======plot track=================")
                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )
                    #predicted_bbox = torch.tensor(np.array(predicted_bbox), dtype=torch.float32)

                else:
                    online_im = img_info['raw_img']


            else:
                # for mota purpose
                if frame_id not in detection_result.keys():
                    detection_result[frame_id]={}

                print(predicted_bbox)
                if len(predicted_bbox) != 0:
                    print(predicted_bbox[0][7])
                    #print(f"bb: {predicted_bbox.shape}")
                    print(f"online target: {online_targets}")

                    light_track_ok, light_track_bbox = light_multi_tracker.update(frame)
                    predict_bbox = []
                    kf_adj = 1  ## keep 1.1 for virat_001 
                    for idx, each_track in enumerate(online_targets):
                        # check if the bbox is in the light_tracker_list -> new track or reinitiated track 
                        
                        if light_track_ok and each_track.track_id in light_tracker_id:
                            if each_track.track_id in light_tracker_id:
                                light_id = light_tracker_id.index(each_track.track_id)
                                new_bbox = light_track_bbox[light_id]
                                # need to convert from tlwh to tlbr
                                predict_bbox.append([new_bbox[0], new_bbox[1], new_bbox[2]+new_bbox[0], new_bbox[3]+new_bbox[1], predicted_bbox[idx][4], predicted_bbox[idx][5]])
                            else:
                                

                                new_x = each_track.mean[0]+each_track.mean[4]/kf_adj
                                new_y = each_track.mean[1]+each_track.mean[5]/kf_adj
                                new_a = each_track.mean[2]+each_track.mean[6]/kf_adj
                                new_h = each_track.mean[3]+each_track.mean[7]/kf_adj
                                new_w = new_a * new_h
                                tlwh = [new_x - new_w/2, new_y - new_h/2, new_w, new_h]

                                # converting the predicted tlwh to tlbr and use this as the new detection bbox
                                tlbr = np.asarray(tlwh).copy()
                                tlbr[2:] += tlbr[:2]
                                tlbr=np.append(tlbr,[predicted_bbox[idx][4], predicted_bbox[idx][5]])
                                predict_bbox.append(tlbr)
                        else:
                            new_x = each_track.mean[0]+each_track.mean[4]/kf_adj
                            new_y = each_track.mean[1]+each_track.mean[5]/kf_adj
                            new_a = each_track.mean[2]+each_track.mean[6]/kf_adj
                            new_h = each_track.mean[3]+each_track.mean[7]/kf_adj
                            new_w = new_a * new_h
                            tlwh = [new_x - new_w/2, new_y - new_h/2, new_w, new_h]

                            # converting the predicted tlwh to tlbr and use this as the new detection bbox
                            tlbr = np.asarray(tlwh).copy()
                            tlbr[2:] += tlbr[:2]
                            tlbr=np.append(tlbr,[predicted_bbox[idx][4], predicted_bbox[idx][5]])
                            predict_bbox.append(tlbr)
                            
                    print(f"after processsing :{predict_bbox}")

                    

                    # if light_track_ok:
                    #     bbox_i=0
                    #     for each in light_tracker_id:
                    #         new_bbox=light_track_bbox[bbox_i]
                    #         for t in online_targets:
                    #             if t.track_id == each: 
                    #                 t.mean[0] = new_bbox[0] + new_bbox[2]/2 
                    #                 t.mean[1] = new_bbox[1] + new_bbox[3]/2
                    #                 t.mean[2] = new_bbox[2]/new_bbox[3]
                    #                 t.mean[3] = new_bbox[3]
                    #                 bbox_i += 1

                    # use the existing track to predict the location and kalman filter state to update the detection bbox for the next frame
                    # predict_bbox = []
                    # for idx, each_track in enumerate(online_targets):
                    #     print(f"track id: {each_track.track_id} class: {each_track.class_id} ???????")
                    #     print(f"mean: {each_track.mean}")
                    #     kf_adj = 1.03  ## keep 1.1 for virat_001 
                    #     if each_track.mean[2]* each_track.mean[3] > args.min_box_area:
                    #         new_x = each_track.mean[0]+each_track.mean[4]/kf_adj
                    #         new_y = each_track.mean[1]+each_track.mean[5]/kf_adj
                    #         new_a = each_track.mean[2]+each_track.mean[6]/kf_adj
                    #         new_h = each_track.mean[3]+each_track.mean[7]/kf_adj
                    #         new_w = new_a * new_h
                    #         tlwh = [new_x - new_w/2, new_y - new_h/2, new_w, new_h]

                    #         # converting the predicted tlwh to tlbr and use this as the new detection bbox
                    #         tlbr = np.asarray(tlwh).copy()
                    #         tlbr[2:] += tlbr[:2]
                    #         tlbr=np.append(tlbr,[predicted_bbox[idx][4], predicted_bbox[idx][5]])
                    #         predict_bbox.append(tlbr)
                    predicted_bbox = torch.tensor(np.array(predict_bbox), dtype=torch.float32)


                    # using the predicted bbox as the new detection result and feed into the tracker update
                    online_targets = tracker.new_update(predicted_bbox, [img_info['height'], img_info['width']], exp.test_size)
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

                        #vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area: #and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_class_id.append(t.class_id)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                            detection_result[frame_id][t.track_id]=(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3])


                    online_im = plot_tracking(
                        image=img_info['raw_img'], tlwhs=online_tlwhs, obj_ids=online_ids, online_class_id=online_class_id, frame_id=frame_id + 1, fps=fps, scores=online_scores, class_names = cls_names
                    )

                else:
                    online_im = img_info['raw_img']
            
            frame_id +=1

            if args.save_result:
                vid_writer.write(online_im)
                cv2.imwrite(f'/data/xiang/video_collab/para_det_trk/imgs/{frame_id}.png', online_im)
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