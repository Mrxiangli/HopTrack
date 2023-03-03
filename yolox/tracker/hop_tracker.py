import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import cv2
import numba as nb
from numba import jit
import time
from loguru import logger
import xlsxwriter


from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from scipy.stats import wasserstein_distance

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, class_id):

        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.class_id = int(class_id)
        self.score = score
        self.tracklet_len = 0
        self.color_dist = None
        self.dist_threshold = None
        self.last_detected_frame = None         # track the lastest frame id the object is detected
        self.last_frame = None                  # keep a copy of the most recent detection
        self.paired = False                     # check if a newly created track is paired with an old one
        self.tmp_active = False                 

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self._tlwh = new_track._tlwh.copy()
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        """
        self.frame_id = frame_id
        self.tracklet_len += 1


        self._tlwh = new_track._tlwh.copy()
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track._tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.class_id = int(new_track.class_id)
        self.score = new_track.score
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class HOPTracker(object):
    def __init__(self, args, frame_rate=30):
        self.args = args
        self.tracked_stracks = []   # type: list[STrack], include all tracks thats not removed
        self.lost_stracks = []      # type: list[STrack], include lost tracks
        self.removed_stracks = []   # type: list[STrack], include tracks removed 
        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.fuse_ct = 0
        self.fuse_time = 0
        if args.source != "webcam":
            self.workbook = xlsxwriter.Workbook(f"{args.path.split('/')[-1].split('.')[0]}"+"_time.xlsx")
            self.worksheet = self.workbook.add_worksheet()
            self.worksheet.write(0, 0, "detect_fuse")
            self.worksheet.write(0, 1, "post_detect")
            self.worksheet.write(0, 2, "track_fuse")
            self.worksheet.write(0, 3, "post_track")

    def detect_track_fuse(self, output_results, img_info, img_size):
        self.frame_id += 1
        activate_tracks = []            # keep track of online tracks
        refind_stracks = []             # keep track of reactivatede tracks
        lost_stracks = []               # keep track of lost tracks
        removed_stracks = []            # keep track of removed 


        start_time = time.time()
        # output result are the most recent detection result from DNN
        output_results = output_results.cpu().numpy()
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        cls_name = output_results[:,5]

        img_h, img_w, frame = img_info[0], img_info[1], img_info[2]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        
        if "yolox" in self.args.model:
            bboxes /= scale

        # detection with high scores -> usually means clear view, no occulsion
        remain_inds = scores > self.args.track_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        class_keep = cls_name[remain_inds]

        # create a track for every single detection
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, class_keep)]
        else:
            detections = []
        
         # keep track of the activated tracks in the current fusion
        tracked_tracks = []        

        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_tracks.append(track)

        """ Similar to Bytetrack, perform one high detection match"""
        # IoU based association 
        strack_pool = joint_stracks(tracked_tracks, self.lost_stracks)

        # Predict the current location with KF and perform linear assignment
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.6)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet] 
            # update the ROI copy for potential future check
            area_of_interest = frame[int(det.tlwh[1]):int(det.tlwh[1]+det.tlwh[3]), int(det.tlwh[0]):int(det.tlwh[0]+det.tlwh[2])]
            track.last_frame = area_of_interest
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activate_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
    
        """Initiating temporary new tracks for unmatched detection objects, perform trajectory finding"""

        for inew in u_detection:
            track = detections[inew]
            if track.score < 0.35:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            area_of_interest = frame[int(track.tlwh[1]):int(track.tlwh[1]+track.tlwh[3]), int(track.tlwh[0]):int(track.tlwh[0]+track.tlwh[2])]
            track.last_frame = area_of_interest
            track.tmp_active = True
        
        # tmp_active contains the track created on new detections, thats not matched in the beigining
        tmp_active = [detections[inew] for inew in u_detection if detections[inew].tmp_active]
            
        """ matching new track with fast moving object"""
        # find the unmatched tracks
        rem_tracks = [strack_pool[i] for i in u_track]

        # unmatched after pixel based finder
        was_unmatched_track = []
        was_unmatched_det = []

        match_start = time.time()
        # the cases that the IOU falling below the threshold
        detections = tmp_active                                  
        dist = matching.iou_distance(rem_tracks, tmp_active)
        matches, u_track, u_detection = matching.linear_assignment(dist, thresh=0.8)

        for itracked, idet in matches:
            track = rem_tracks[itracked]
            det = tmp_active[idet] 
            # update the ROI copy for potential future check
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activate_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        was_unmatched_track = [rem_tracks[i] for i in u_track]
        was_unmatched_det = [detections[i] for i in u_detection]
        
        # """switch here"""
        # if dist.size != 0:
        #     tmp_match = list(np.argmin(dist,axis=1))        # this tmp match is trying to mapping the temporary new tracks with remaining tracks
        #     for idx, each in enumerate(tmp_match):
        #         if dist[idx][each]==1:
        #             tmp_match[idx]=-1

        #     matched_det = []
        #     for idx, each in enumerate(tmp_match):
        #         if each != -1 :
        #             # the detections are new tracks in this case, the last 4 states of KF are initialized
        #             if abs(detections[each].mean[4]<0.0001) and abs(detections[each].mean[5]<0.0001) and abs(detections[each].mean[6]<0.0001) and abs(detections[each].mean[7]<0.0001):

        #                 ot_x, ot_y, ot_w, ot_h = rem_tracks[idx]._tlwh
        #                 last_detected_frame = rem_tracks[idx].last_detected_frame
        #                 if np.any(rem_tracks[idx]._tlwh<0):
        #                     continue
        #                 d_x,d_y, dw, dh = detections[each]._tlwh
        #                 if np.any(detections[each]._tlwh<0):
        #                     continue
        #                 d_y_max = img_h if d_y+dh > img_h else d_y+dh
        #                 d_x_max = img_w if d_x+dw > img_w else d_x+dw
        #                 """
        #                 The following code perform image quantization and analysis, the quantization patches can be tuned for each video
        #                 Later on, this can be updated respect to the obejct aspect ratio. The setting in this version can produce 61.56 on MOT16
        #                 """
        #                 det_roi = frame[int(d_y):int(d_y_max), int(d_x):int(d_x_max)]
        #                 # adjust the birghtness
        #                 det_roi = np.array(det_roi/(det_roi.mean()/rem_tracks[idx].last_frame.mean()),dtype=np.uint8)
        #                 det_roi[det_roi>255] = 255
        #                 # perform quantization
        #                 quant_list_a = dynamic_chunks(det_roi,4,3)
        #                 quant_list_b = dynamic_chunks(rem_tracks[idx].last_frame,4,3)

        #                 image_partition_list_a = quantization(det_roi,quant_list_a,4,3)
        #                 image_partition_list_b = quantization(rem_tracks[idx].last_frame,quant_list_b,4,3)
        #                 # calculation wasserstein distance for each tile
        #                 WASS=my_wass(image_partition_list_a, image_partition_list_b, 4,3)
        #                 # perform majority vote
        #                 if sum(WASS.flatten()<=0.20)>=6:
        #                     tmp_id = detections[each].track_id
        #                     detections[each].track_id = rem_tracks[idx].track_id
        #                     #print(f"track {detections[each].track_id} is changed to track {rem_tracks[idx].track_id}")
        #                     rem_tracks[idx].track_id = tmp_id
        #                     area_of_interest = det_roi
        #                     detections[each].last_frame = area_of_interest
                            
        #                     # since the high speed objects and the new detection has low ious, it was treated as two seperated objects initially and both will has 0 for last 4 kalman states,
        #                     nd_x, nd_y, nd_w, nd_h = detections[each]._tlwh
        #                     old_center_x = ot_x + ot_w/2
        #                     old_center_y = ot_y + ot_h/2
        #                     new_center_x = nd_x + nd_w/2
        #                     new_center_y = nd_y + nd_h/2
        #                     new_vx = (new_center_x - old_center_x)/(self.frame_id - last_detected_frame) # divided by the lost number of frames: could reinit last 4 kf state with 0, so it reinit to medianflow
        #                     new_vy = (new_center_y - old_center_y)/(self.frame_id - last_detected_frame) # 
        #                     detections[each].mean[0] =  new_center_x
        #                     detections[each].mean[1] =  new_center_y
        #                     detections[each].mean[2] =  dw/dh
        #                     detections[each].mean[3] =  dh
        #                     detections[each].mean[4] = (rem_tracks[idx].mean[4]+new_vx)/2 
        #                     detections[each].mean[5] = (rem_tracks[idx].mean[5]+new_vy)/2
        #                     detections[each].mean[6] =  rem_tracks[idx].mean[6]
        #                     detections[each].mean[7] =  rem_tracks[idx].mean[7]


        #                     activate_tracks.append(detections[each])
        #                     rem_tracks[idx].mark_removed()
        #                     #print(f"track {rem_tracks[idx].track_id} is removed")
        #                     matched_det.append(detections[each])
        #                     detections[each].paired = True
        #                 else:
        #                     was_unmatched_track.append(rem_tracks[idx])
        #                     #print(f"track {rem_tracks[idx].track_id} is added to unmatch")
        #             else:
        #                 was_unmatched_track.append(rem_tracks[idx])
                        
        #         else:
        #             was_unmatched_track.append(rem_tracks[idx])
                     
            
        #    # if not a single match (not even slightly IOU)

        #     for idx, each in enumerate(tmp_match):          
        #         if each == -1:
        #             track = rem_tracks[idx]
        #             if track not in was_unmatched_track:
        #                 was_unmatched_track.append(track)

        #     for idx, each in enumerate(tmp_active):  #if not in the match, no IoU at all
        #         if each.paired == False:
        #             was_unmatched_det.append(each)   

        # else:
        #     # if no match which means all the newly not necessary
        #     for idx, each in enumerate(tmp_active):
        #        was_unmatched_det.append(each) 
            
        #     # if not match, all remaining track are lost tracks
        #     for each in rem_tracks: 
        #         if not each.state == TrackState.Lost:
        #             was_unmatched_track.append(each)
                          
        # """ switch here"""
        # logger.info("matching time: {:.4f}ms".format((time.time() - match_start)*1000))
        if self.args.dis_traj == False:
            """trajectory based finding, existing traj with KF state"""
            traj_KF = []
            for each in was_unmatched_track:
                if each.tracklet_len > 90 or self.frame_id < 60:  # the object is tracked more than 30 frames, so the KF should be fairly stable, or at the beginning of the video
                    if np.any(each._tlwh<=0):
                        continue
                    traj_KF.append(each)

            matched_det_traj = []
            matched_track_traj = []

            for each_track in traj_KF:
                tmp_rank = []
                for idx, each_det in enumerate(was_unmatched_det):
                    distance = trajectory_finder(each_track, each_det)
                    tmp_rank.append((distance, idx))
                # sort based on distance rom low to high
                tmp_rank.sort(key=lambda tmp_rank: tmp_rank[0])       

                for i in range(len(tmp_rank)):
                    if was_unmatched_det[tmp_rank[i][1]].paired == False:
                        potential_match = was_unmatched_det[tmp_rank[i][1]]
                        d_x, d_y, dw, dh = potential_match._tlwh

                        if np.any(potential_match._tlwh<=0):
                            continue

                        d_y_max = img_h if d_y+dh > img_h else d_y+dh
                        d_x_max = img_w if d_x+dw > img_w else d_x+dw
                        det_roi = frame[int(d_y):int(d_y+dh), int(d_x):int(d_x+dw)]

                        det_roi = np.array(det_roi/(det_roi.mean()/each_track.last_frame.mean()),dtype=np.uint8)
                        det_roi[det_roi>255] = 255
                        quant_list_a = dynamic_chunks(det_roi,7,3)
                        quant_list_b = dynamic_chunks(each_track.last_frame,7,3)

                        image_partition_list_a = quantization(det_roi,quant_list_a,7,3)
                        image_partition_list_b = quantization(each_track.last_frame,quant_list_b,7,3)
                        WASS=my_wass(image_partition_list_a, image_partition_list_b, 7,3)

                        if sum(WASS[:,1]<=0.20)>=4:
                            # if len(tmp_rank) >= 3:
                            #     plot_potential_match(self.frame_id, frame, was_unmatched_det, each_track, tmp_rank)
                            tmp_id = potential_match.track_id
                            potential_match.track_id = each_track.track_id
                            potential_match.track_id = tmp_id
                            potential_match.last_frame = det_roi
                            # sending it in to median flow for rein
                            each_track.mean[4] = 0
                            each_track.mean[5] = 0
                            each_track.mean[6] = 0
                            each_track.mean[7] = 0
                            each_track.mark_removed()
                            removed_stracks.append(each_track)
                            activate_tracks.append(potential_match)

                            matched_det_traj.append(was_unmatched_det[tmp_rank[i][1]])
                            matched_track_traj.append(each_track)
                            break
                        if i > 2:
                            break

        
        for each in was_unmatched_det:                      # true new objects
            if each.paired == False:
                activate_tracks.append(each)

        for each in was_unmatched_track: 
            if self.args.dis_traj == False:         
                if not each.state == TrackState.Lost and each not in matched_track_traj :
                    each.mark_lost()
                    lost_stracks.append(each)
            else:
                if not each.state == TrackState.Lost :
                    each.mark_lost()
                    lost_stracks.append(each)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        if (time.time()-start_time)*1000 < 1000:
            self.fuse_time += (time.time()-start_time)*1000
            self.fuse_ct += 1

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activate_tracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if (track.is_activated and (track.class_id == 0 or track.class_id == 1 or track.class_id == 2))] 

        duration = (time.time() - start_time)*1000
        #logger.info("fuse time: {:.4f}ms".format(duration))
        if self.args.source != "webcam":
            self.worksheet.write(self.frame_id, 0, duration)
        return output_stracks

    def hopping_update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activate_tracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        hop_start = time.time()

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        cls_name = output_results[:,5]

        img_h, img_w, frame = img_info[0], img_info[1], img_info[2]

        inds_high = scores >= 0.3   #self.args.track_thresh
        
        #detection with high scores
        dets = bboxes[inds_high]
        scores_keep = scores[inds_high]
        class_keep = cls_name[inds_high]

        

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, class_keep)]
        else:
            detections = []

        tracked_tracks = []  # type: list[STrack]

        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_tracks.append(track)
        
        # association based on kalman filter predicted position, theorectically they should align
        strack_pool = joint_stracks(tracked_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.6)

        # associate the tracks in the high score detection with the existing tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activate_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if self.args.dis_traj == False:
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            unmatched_detections = detections

            potentially_lost = []
            detection_occulusion = []
            u_track = []
            u_detectons = []

            new_thresh = 0.8
            dists = matching.iou_distance(r_tracked_stracks, unmatched_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=new_thresh)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = unmatched_detections[idet]
                d_x, d_y, dw, dh = det._tlwh

                det_roi = frame[int(d_y):int(d_y+dh), int(d_x):int(d_x+dw)]
                det_roi = np.array(det_roi/(det_roi.mean()/track.last_frame.mean()),dtype=np.uint8)
                det_roi[det_roi>255] = 255
                t_x, t_y, t_w, t_h = track._tlwh
                
                quant_list_a = dynamic_chunks(det_roi,6,3)
                quant_list_b = dynamic_chunks(track.last_frame,6,3)

                image_partition_list_a = quantization_color_vector(det_roi,quant_list_a,6,3)
                image_partition_list_b = quantization_color_vector(track.last_frame,quant_list_b,6,3)
                cost=color_vector_cost_calculation(image_partition_list_a, image_partition_list_b, 6,3)

                color_matches, color_u_track, color_u_detection = matching.linear_assignment(cost, thresh=0.0001)

                if color_matches.shape[0] >= 9:
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activate_tracks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                else:
                    potentially_lost.append(itracked)
                    detection_occulusion.append(idet)


            lost_stracks = []
            for idx in u_track:

                track = r_tracked_stracks[idx]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track) 
            
            for idx in potentially_lost:
                track = r_tracked_stracks[idx]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        if self.args.dis_traj == True:
            lost_stracks = []
            for idx in u_track:

                track = strack_pool[idx]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track) 


        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activate_tracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
      
        output_stracks = [track for track in self.tracked_stracks if (track.is_activated and (track.class_id == 0 or track.class_id == 1 or track.class_id == 2))]
        duration = (time.time() - hop_start)*1000
        #logger.info("hop time: {:.4f}ms".format(duration))
        if self.args.source != "webcam":
            self.worksheet.write(self.frame_id, 2, duration)
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlistb:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlista:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

@jit
def trajectory_finder(track, det):
    # get an hint of the moving direction
    tl_x, tl_y, width, height = track._tlwh
    vx = track.mean[4]
    vy = track.mean[5]

    dl_x, dl_y, width_d, height_d = det._tlwh

    # center of the track obejct
    c_track_x = tl_x + width/2
    c_track_y = tl_y + height/2

    # one step further
    c_track_x_next = tl_x + width/2 + vx
    c_track_y_next = tl_y + height/2 + vy

    # det center calculation

    dl_x_c = dl_x + width_d/2
    dl_y_c = dl_y + height_d/2

    # calculate norm from det to the line connects c_track and c_track_next

    p_det = np.array([dl_x_c, dl_y_c])
    p_track = np.array([c_track_x, c_track_y])
    p_track_next = np.array([c_track_x_next, c_track_y_next])
    if abs(p_track_next[0] - p_track[0]) <= 1e-5 and abs(p_track_next[1]-p_track[1])<=1e-5:
        distance = np.linalg.norm(p_track_next - p_det)
    else:
        distance = np.linalg.norm(np.cross(p_track_next - p_track, p_track - p_det))/np.linalg.norm(p_track_next - p_track)

    return distance

@jit
def dynamic_chunks(img,row,col):
    height, width = img.shape[:2]
    row_interval = int(height/row)
    col_interval = int(width/col)
    quantization_list = np.empty((row,col),dtype=object)
    for i in range(row):
        for j in range(col):
            quantization_list[i][j] = (i*row_interval, j*col_interval, col_interval, row_interval)
    return quantization_list 

@jit
def quantization(img,quant_list,row,col):
    height, width = img.shape[:2]

    image_partition_list = np.empty((row,col),dtype=object)
    for i in range(row):
        for j in range(col):
            y, x, w, h = quant_list[i][j]
            top_x = x
            top_y = y
            bot_x = top_x + w
            bot_y = top_y + h
            if bot_x >= width:
                bot_x = width-1
            if bot_y >= height:
                bot_y = height-1
            img_partition = img[top_y:bot_y, top_x:bot_x]
            image_partition_list[i][j] = img_partition
    return image_partition_list

@jit
def my_wass(quan_a, quan_b, row, col):
    wass_matrix = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            new_img_a = quan_a[i][j]
            
            hist_b_a = cv2.calcHist([new_img_a],[0],None,[256],[0,256])       # blue channel
            hist_g_a = cv2.calcHist([new_img_a],[1],None,[256],[0,256])       # green channel
            hist_r_a = cv2.calcHist([new_img_a],[2],None,[256],[0,256])       # red channel

            cv2.normalize(hist_b_a,hist_b_a,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_g_a,hist_g_a,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_r_a,hist_r_a,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_b_a = hist_b_a.flatten()
            hist_g_a = hist_g_a.flatten()
            hist_r_a = hist_r_a.flatten()

            new_img_b = quan_b[i][j]
            
            hist_b_b = cv2.calcHist([new_img_b],[0],None,[256],[0,256])       # blue channel
            hist_g_b = cv2.calcHist([new_img_b],[1],None,[256],[0,256])       # green channel
            hist_r_b = cv2.calcHist([new_img_b],[2],None,[256],[0,256])       # red channel

            cv2.normalize(hist_b_b,hist_b_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_g_b,hist_g_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_r_b,hist_r_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_b_b = hist_b_b.flatten()
            hist_g_b = hist_g_b.flatten()
            hist_r_b = hist_r_b.flatten()

            b_s = wasserstein_distance(hist_b_a, hist_b_b, hist_b_a, hist_b_a)
            g_s = wasserstein_distance(hist_g_a, hist_g_b, hist_g_a, hist_g_a)
            r_s = wasserstein_distance(hist_r_a, hist_r_b, hist_r_a, hist_r_a)
            wass_matrix[i][j] = (b_s + g_s + r_s)/3
    return wass_matrix

@jit
def pixel_distribution(img, tl_x, tl_y, width, height):
    new_img = img[tl_y:tl_y+height, tl_x:tl_x+width]
    hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel
    hist_g = cv2.calcHist([new_img],[1],None,[256],[0,256])       # green channel
    hist_r = cv2.calcHist([new_img],[2],None,[256],[0,256])       # red channel

    cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_g,hist_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_r,hist_r,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return (hist_b.flatten(), hist_g.flatten(), hist_r.flatten())

@jit
def quantization_color_vector(img,quant_list,row,col):
    height, width = img.shape[:2]
    img = img/255
    image_partition_list = np.empty((row,col),dtype=object)
    for i in range(row):
        for j in range(col):
            y, x, w, h = quant_list[i][j]
            top_x = x
            top_y = y
            bot_x = top_x + w
            bot_y = top_y + h
            if bot_x >= width:
                bot_x = width-1
            if bot_y >= height:
                bot_y = height-1
            img_partition = img[top_y:bot_y, top_x:bot_x]
            image_partition_list[i][j] = img_partition
    return image_partition_list

def color_vector_cost_calculation(quan_a, quan_b, row, col):
    cost_matrix = np.zeros((row*col, row*col))

    for i_a in range(row):
        for j_a in range(col):
            
            new_img_a = quan_a[i_a][j_a]
            pix_a = new_img_a.shape[0]*new_img_a.shape[1]
                    
            suma_b = np.sum(new_img_a[:,:,0])/pix_a      # blue channel
            suma_g = np.sum(new_img_a[:,:,1])/pix_a      # green channel
            suma_r = np.sum(new_img_a[:,:,2])/pix_a      # red channel
            
            a_color = np.array([suma_b, suma_g, suma_r])

            for i_b in range(row):
                for j_b in range(col):

                    new_img_b = quan_b[i_b][j_b]
                    pix_b = new_img_b.shape[0]*new_img_b.shape[1]
                    
                    sumb_b = np.sum(new_img_b[:,:,0])/pix_b      # blue channel
                    sumb_g = np.sum(new_img_b[:,:,1])/pix_b      # green channel
                    sumb_r = np.sum(new_img_b[:,:,2])/pix_b      # red channel
                    # print(sumb_b)
                    # print(sumb_g)
                    # print(sumb_r)
                    if sumb_b ==0 and sumb_g ==0 and sumb_r ==0:
                        print(new_img_b)
                        import sys
                        sys.exit()

                    b_color =  np.array([sumb_b, sumb_g, sumb_r])

                    dis = 1 - np.dot(a_color, b_color)/(np.linalg.norm(a_color)*np.linalg.norm(b_color))

                    cost_matrix[i_a*col+j_a][i_b*col+j_b] = dis
                   
    return cost_matrix

"""
Use the following functions to visualize tracks, matches, or plot the trajctory finder match on the frame 
"""
def track_listing(tk_list,collection_name):
    print(f"The tracks in {collection_name} are listed below -------")
    if len(tk_list) != 0:
        for each in tk_list:
            print(f"{each.track_id},{each.tlwh[0]},{each.tlwh[1]},{each.tlwh[2]},{each.tlwh[3]}, latest det: {each.last_detected_frame}")  

def tmp_match_print(tp_list, tk, det):
    for idx, each in enumerate(tp_list):
        if each == -1:
            print(f"track {tk[idx].track_id}: None {tk[idx].last_frame.shape}")

        else:
            print(f"track {tk[idx].track_id} : {tk[idx]._tlwh} {det[each]._tlwh}")

def plot_potential_match(frame_id, frame, det, orig_track, tmp_rank):
    import copy
    img = copy.deepcopy(frame)
    ct = 0
    for each in tmp_rank:
        roi = det[each[1]]._tlwh
        tx  = int(roi[0])
        ty = int(roi[1])
        bx = int(roi[0]+roi[2])
        by = int(roi[1]+roi[3])
        img_path_det = os.getcwd()+"frame_"+str(frame_id)+"track_"+str(orig_track.track_id)+str(ct)+"_det.png"
        cv2.imwrite(img_path_det, img[ty:by, tx:bx])
        img = cv2.rectangle(img, (tx, ty), (bx, by), (255,255,0), 2)
        ct+=1
        if ct==3:
            break
        img = copy.deepcopy(frame)
    
    roi = orig_track._tlwh
    tx  = int(roi[0])
    ty = int(roi[1])
    bx = int(roi[0]+roi[2])
    by = int(roi[1]+roi[3])
    img = cv2.rectangle(img, (tx, ty), (bx, by), (0,255,255), 2)
    img_path = os.getcwd()+"frame_"+str(frame_id)+"track_"+str(orig_track.track_id)+".png"
    cv2.imwrite(img_path, img)
    img_path_det = os.getcwd()+"frame_"+str(frame_id)+"track_"+str(orig_track.track_id)+"_det.png"
    cv2.imwrite(img_path_det, orig_track.last_frame)

    
