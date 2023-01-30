import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import cv2

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from scipy.stats import wasserstein_distance


def pixel_distribution(img, tl_x, tl_y, width, height):
    new_img = img[tl_y:tl_y+height, tl_x:tl_x+width]
    hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel
    hist_g = cv2.calcHist([new_img],[1],None,[256],[0,256])       # green channel
    hist_r = cv2.calcHist([new_img],[2],None,[256],[0,256])       # red channel

    cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_g,hist_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_r,hist_r,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return (hist_b.flatten(), hist_g.flatten(), hist_r.flatten())

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
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
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
        self.tracked_stracks = []   # type: list[STrack], include all tracks thats not removed
        self.lost_stracks = []      # type: list[STrack], include lost tracks
        self.removed_stracks = []   # type: list[STrack], include tracks removed 

        self.frame_id = 0
        self.args = args
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()


    def detect_track_fuse(self, output_results, img_info, img_size):
        self.frame_id += 1
        activate_tracks = []            # keep track of online tracks
        refind_stracks = []             # keep track of reactivatede tracks
        lost_stracks = []               # keep track of lost tracks
        removed_stracks = []            # keep track of removed tracks
   
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

        unconfirmed = []
        tracked_tracks = []         # keep track of the activated tracks in the current fusion

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)

        # IoU based association -> may run wasserstain distance check to make sure no cross association happened
        strack_pool = joint_stracks(tracked_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # associate the tracks in the high score detection with the existing tracks
        unmatch_high_det = []
        unmatch_high_track = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            top_x, top_y, tw, th = det._tlwh
            hist_b, hist_g, hist_r = pixel_distribution(frame, int(top_x), int(top_y), int(tw), int(th))
            wass_b = wasserstein_distance( hist_b, track.color_dist[0])
            wass_g = wasserstein_distance( hist_g, track.color_dist[1])
            wass_r = wasserstein_distance( hist_r, track.color_dist[2])

            if (wass_b + wass_g + wass_r)/3 < 0.5:
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activate_tracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            else:
                unmatch_high_det.append(det)
                unmatch_high_track.append(track)
    
        """ Need to run an additional wassertain distance for unmatched object from high confidence: the case that involved large displacement"""


        """ not sure if this is needed, need further testing """
        detections = [detections[i] for i in u_detection]
        for each in unmatch_high_det:
            detections.append(each)
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activate_tracks.append(unconfirmed[itracked])

        #before removing the unconfirmed track, try to perform an analysis of wassertein distance
        detection_his = []
        track_his= []
        for it in u_detection:
            u_det = detections[it]
            top_x, top_y, tw, th = u_det._tlwh
            hist_b, hist_g, hist_r = pixel_distribution(frame, int(top_x), int(top_y), int(tw), int(th))
            detection_his.append([hist_b, hist_g, hist_r])
        
        """shouldn't use the current frame for the wassestin distance, should be the previous detection one"""
        for it in u_unconfirmed:
            unconfirmed_track = unconfirmed[it]
            top_x, top_y, tw, th = unconfirmed_track._tlwh
            hist_b, hist_g, hist_r = pixel_distribution(frame, int(top_x), int(top_y), int(tw), int(th))
            track_his.append([hist_b, hist_g, hist_r])

        matching_pair = []
        for i, d_his in enumerate(detection_his):
            tmp_match = -1
            for j, t_his in enumerate(track_his):
                tmp_best = 0
                wass_b = wasserstein_distance(d_his[0], t_his[0])
                wass_g = wasserstein_distance(d_his[1], t_his[1])
                wass_r = wasserstein_distance(d_his[2], t_his[2])
                if (wass_b + wass_g + wass_r)/3 < 0.5 and (wass_b + wass_g + wass_r)/3 > tmp_best:
                    tmp_match = j
                    tmo_best = (wass_b + wass_g + wass_r)/3
            matching_pair.append((i,tmp_match))

        for each_pair in matching_pair:
            if each_pair[1] != -1:
                unconfirmed[each_pair[0]].update(detections[each_pair[1]], self.frame_id)
                u_unconfirmed.pop(each_pair[0])     
                u_detection.pop(each_pair[1])

        # for remaining unmatched tracks, remove them from the tracks    

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        # initiating new tracks for new objects
        tmp_active=[]
        for inew in u_detection:
            track = detections[inew]
            if track.score < 0.35:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            tmp_active.append(track)
            

        """ matching new track with fast moving object"""
        # matching the fast moving objects:
        # find the un matched tracks
        rem_tracks = [strack_pool[i] for i in u_track]
        for each in unmatch_high_track:
            rem_track.append(each)

        # find the unmatched detections with high detection scores
        detections = tmp_active                                     # detections not matched with IoU or unconfirmed
        dist = matching.iou_distance(rem_tracks, tmp_active)
        if dist.size != 0:
            tmp_match = list(np.argmin(dist,axis=1))        # this tmp match is trying to mapping the temporary new tracks with remaining tracks
            for idx, each in enumerate(tmp_match):
                if dist[idx][each]==1:
                    tmp_match[idx]=-1
            for idx, each in enumerate(tmp_match):
                if each != -1:
                    # the detections are new tracks in this case, the last 4 states of KF are initialized
                    if abs(detections[each].mean[4]<0.0001) and abs(detections[each].mean[5]<0.0001) and abs(detections[each].mean[6]<0.0001) and abs(detections[each].mean[7]<0.0001):
                        ot_x, ot_y, ot_w, ot_h = rem_tracks[idx]._tlwh
                        rem_tracks[idx].update(detections[each],self.frame_id)
                        # since the high speed objects and the new detection has low ious, it was treated as two seperated objects initially and both will has 0 for last 4 kalman states,
                        nd_x, nd_y, nd_w, nd_h = detections[each]._tlwh
                        old_center_x = ot_x + ot_w/2
                        old_center_y = ot_y + ot_h/2
                        new_center_x = nd_x + nd_w/2
                        new_center_y = nd_y + nd_h/2
                        new_vx = (new_center_x - old_center_x) # cuz we have 10 frames in the middle, need to change correspondingly
                        new_vy = (new_center_y - old_center_y) # keep 7 for the traffic video test
                        rem_tracks[idx].mean[4] = new_vx  
                        rem_tracks[idx].mean[5] = new_vy
                        activate_tracks.append(rem_tracks[idx])

            # mark the rest of tracks as lost tracks, need to move to later section
            for idx, each in enumerate(tmp_match):          
                if each == -1:
                    track = rem_tracks[idx]
                    if not track.state == TrackState.Lost:
                        track.mark_lost()
                        lost_stracks.append(track)

            for idx, each in enumerate(tmp_active):  #if not in the match, which means its a true new objects
                if idx not in tmp_match or len(tmp_match)==0:
                    activate_tracks.append(each)
        else:
            # if no match which means all the newly created tracks are new objects
            for idx, each in enumerate(tmp_active):
                activate_tracks.append(each)
            
            # if not match, all remaing track are lost tracks
            for each in rem_tracks:          
                if not each.state == TrackState.Lost:
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
        
        return output_stracks

    def hopping_update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activate_tracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

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

        unconfirmed = []
        tracked_tracks = []  # type: list[STrack]

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)

        # association based on kalman filter predicted position, theorectically they should align
        strack_pool = joint_stracks(tracked_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

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

        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        unmatched_detections = detections# + detections_second

        potentially_lost = []
        detection_occulusion = []
        new_thresh=0.1
        u_track = []
        u_detectons = []
        while len(r_tracked_stracks)>0 and len(unmatched_detections)>0 and new_thresh > 0.1:
            new_thresh -= 0.1
            dists = matching.iou_distance(r_tracked_stracks, unmatched_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=new_thresh)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = unmatched_detections[idet]
                t_x, t_y, t_w, t_h = track._tlwh
                d_x, d_y, d_w, d_h = det._tlwh
                t_hist_b, t_hist_g, t_hist_r = pixel_distribution(frame, int(t_x), int(t_y), int(t_w), int(t_h))
                d_hist_b, d_hist_g, d_hist_r = pixel_distribution(frame, int(d_x), int(d_y), int(d_w), int(d_h))
                wass_b = wasserstein_distance(t_hist_b, d_hist_b)
                wass_g = wasserstein_distance(t_hist_g, d_hist_g)
                wass_r = wasserstein_distance(t_hist_r, d_hist_r)
                average_wass = (wass_b + wass_g + wass_r)/3

                if average_wass < 0.5:
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activate_tracks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                else:
                    potentially_lost.append(itrack)
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

