import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, class_id):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.class_id = int(class_id)
        self.score = score
        self.tracklet_len = 0
        self.first_seen = True
        self.kalman_adjust = False
        self.kalman_adjust_period = 5
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
            #print(f"multi_mean: {multi_mean}")

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        #print(f"mean after kalman init: {self.mean}")

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #if frame_id == 1:
        #    self.is_activated = True
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
        # this is updated by Xiang Li
        self._tlwh = new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.class_id = int(new_track.class_id)
        self.score = new_track.score

    @property
    # @jit(nopython=True)
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
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
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
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 6:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            cls_name = output_results[:,5]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            cls_name = output_results[:,6]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        ## detection with high scores
     #   print(f"scores: {scores}")
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.4
        inds_high = scores < self.args.track_thresh
     #   print(f"remain {remain_inds}")
      #  print(f"low {inds_low}")
     #   print(f"inds_high {inds_high}")
        
        #detection with low scores
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        class_keep = cls_name[remain_inds]
        scores_second = scores[inds_second]
        class_second = cls_name[inds_second]


        #create track for all detections:
        # objects_all = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for (tlbr, s, c) in zip(bboxes, scores, cls_name)]
        # STrack.multi_predict(objects_all)

        if len(dets) > 0:
            '''Detections: bounding boxes of objects that has a score > track thresh'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, class_keep)]
        else:
            detections = []
     #   print(f"high score detections: {detections}")

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
     #   print(f"self.tracked: {self.tracked_stracks}")
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
      #  print(f"unconfirmed tracks: {unconfirmed}")

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
     #   print(f"s_pool: {strack_pool}")
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
     #   print(f"dists: {dists}")
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
     #   print(f"1st match {matches}")
     #   print(f"1st utrack {u_track}")
     #   print(f"1st u_detections {u_detection}")

        # associate the tracks in the high score detection with the existing tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
            #    print(f"track {track} is reactivated !!!!!!!!!!!!!!!!!")
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets_second, scores_second, class_second)]
        else:
            detections_second = []
     #   print(f"2nd detection: {detections_second}")
        # for unmatched tracks from the first association, if the state is Tracked, then put it in r_tracked_stracks
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
     #   print(f"r_track: {r_tracked_stracks}")
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
     #   print(f"2nd match {matches}")
     #   print(f"2nd utrack {u_track}")
     #   print(f"2nd u_detections {u_detection_second}")
        # assocated the lower score detection with the rest of tracks
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
             #   print(f"track {track} is reactivated !!!!!!!!!!!!!!!!!")
                refind_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        # for remaining unmatched tracks, remove them from the tracks    
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        """ Step 4: Init new stracks"""
        tmp_active=[]
        for inew in u_detection:
         #   print("creating new trackssssssssssssssssss!")
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            tmp_active.append(track)
            
         #   print(f"activated_stracks: {activated_stracks}")

        """ matching new track with fast moving object"""
        # matching the fast moving objects:
        # find the un matched tracks
        rem_tracks = [r_tracked_stracks[i] for i in u_track]
      #  print(f"rem_tracks: {rem_tracks}")
     #   print(f"tmp active: {tmp_active}")
        # find the unmatched detections with high detection scores
        detections = tmp_active 
        dist = matching.iou_distance(rem_tracks, tmp_active)
      #  print(f"dist: {dist} length: {len(dist)} type:{type(dist)}")
        if dist.size != 0:
          #  print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dist: {dist} length: {len(dist)}")
            tmp_match = list(np.argmin(dist,axis=1))
            for idx, each in enumerate(tmp_match):
                if dist[idx][each]==1:
                    tmp_match[idx]=-1
         #   print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!tmp_match: {tmp_match}")
         #   print(f"detections: {detections}")
            for idx, each in enumerate(tmp_match):
                if each != -1:
                  #  print(f"fast track: {rem_tracks[idx]}")
                  #  print(f"fast track mean: {rem_tracks[idx].mean}")
                  #  print(f"corresponding detection: {detections[each]}")
                  #  print(f"corresponding detection mean: {detections[each].mean}")
                    #if abs(rem_tracks[idx].mean[4] < 1) and abs(rem_tracks[idx].mean[5] < 1) and abs(rem_tracks[idx].mean[6] < 1) and abs(rem_tracks[idx].mean[7] < 1):
                    if abs(detections[each].mean[4]<0.0001) and abs(detections[each].mean[5]<0.0001) and abs(detections[each].mean[6]<0.0001) and abs(detections[each].mean[7]<0.0001):
                        #if rem_tracks[idx].class_id == detections[each].class_id:
                      #  print(f"fast track: {rem_tracks[idx]}")
                      #  print(f"corresponding detection: {detections[each]}")
                        ot_x, ot_y, ot_w, ot_h = rem_tracks[idx]._tlwh
                      #  print(f"old track {rem_tracks[idx]._tlwh}")
                        rem_tracks[idx].update(detections[each],self.frame_id)
                        # since the high speed objects and the new detection has low ious, it was treated as two seperated objects initially and both will has 0 for last 4 kalman states,
                        nd_x, nd_y, nd_w, nd_h = detections[each]._tlwh
                      #  print(f"new_detection {detections[each]._tlwh}")
                        old_center_x = ot_x + ot_w/2
                        old_center_y = ot_y + ot_h/2
                        new_center_x = nd_x + nd_w/2
                        new_center_y = nd_y + nd_h/2
                        new_vx = (new_center_x - old_center_x)/9 # cuz we have 10 frames in the middle, need to change correspondingly
                        new_vy = (new_center_y - old_center_y)/9 # keep 7 for the traffic video test
                        #new_vh = (nd_h - ot_h)
                        #new_va = (nd_w/nd_h)/(ot_w/ot_h)
                        rem_tracks[idx].mean[4] = new_vx  
                        rem_tracks[idx].mean[5] = new_vy
                        #rem_tracks[idx].mean[6] = new_vh
                        #rem_tracks[idx].mean[7] = new_va
                        activated_stracks.append(rem_tracks[idx])

            # mark the rest of tracks as lost tracks, need to move to later section
            for idx, each in enumerate(tmp_match):
                if each == -1:
                    track = rem_tracks[idx]
                    if not track.state == TrackState.Lost:
                        track.mark_lost()
                        lost_stracks.append(track)

            for idx, each in enumerate(tmp_active):
                if idx not in tmp_match or len(tmp_match)==0:
                    activated_stracks.append(each)
        else:
            for idx, each in enumerate(tmp_active):
                activated_stracks.append(each)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
      #  print(f"lost tracks: {lost_stracks}")
      #  print(f"removed tracks: {removed_stracks}")

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
      #  print(f"after joining: {self.tracked_stracks}")
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
       # print(f"after joining: {self.tracked_stracks}")
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
      #  print("========================================================")
      #  print(f"output stracks {output_stracks}")
        return output_stracks

    def new_update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 6:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            cls_name = output_results[:,5]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            cls_name = output_results[:,6]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]


        inds_low = scores < self.args.track_thresh
        inds_high = scores >= self.args.track_thresh
        
        #detection with high scores
        dets = bboxes[inds_high]
        scores_keep = scores[inds_high]
        class_keep = cls_name[inds_high]

        ## detection with high scores
        dets_second = bboxes[inds_low]
        scores_second = scores[inds_low]
        class_second = cls_name[inds_low]
        import time
        first_time = time.time()
        if len(dets) > 0:
            '''Detections: bounding boxes of objects that has a score > track thresh'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, class_keep)]
        else:
            detections = []
      #  print(f"high score detections: {detections}")

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
       # print(f"self.tracked: {self.tracked_stracks}")
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
      #  print(f"strack_pool: {strack_pool}")
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        #print(f"first matching: {(time.time()-first_time)*1000}")
      #  print(f"dists: {dists}")
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
       # print(f"1st match {matches}")
       # print(f"1st utrack {u_track}")
      #  print(f"1st u_detections {u_detection}")

        # associate the tracks in the high score detection with the existing tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
               # print("updating the kalman state")
              #  print(f"kalman state before: {track.mean}")
                track.update(det, self.frame_id)
               # print(f"kalman state after: {track.mean}")
                #print(f"updated track: id: {track.class_id}, {track.tlwh}")
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
               # print(f"track {track} is reactivated !!!!!!!!!!!!!!!!!")
                refind_stracks.append(track)
       # print(f"firat association time: {(time.time()-first_time)*1000}")
        second_time = time.time()
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets_second, scores_second, class_second)]
        else:
            detections_second = []
       # print(f"2nd detection: {detections_second}")
        # for unmatched tracks from the first association, if the state is Tracked, then put it in r_tracked_stracks
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
       # print(f"r_track: {r_tracked_stracks}")
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
       # print(f"2nd match {matches}")
       # print(f"2nd utrack {u_track}")
       # print(f"2nd u_detections {u_detection_second}")
        # assocated the lower score detection with the rest of tracks
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                #print("updating the kalman state")
               # print(f"kalman state before: {track.mean}")
                track.update(det, self.frame_id)
               # print(f"kalman state after: {track.mean}")
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
               # print(f"track {track} is reactivated !!!!!!!!!!!!!!!!!")
                refind_stracks.append(track) 
       # print(f"second association time: {(time.time()-second_time)*1000}")

        third_time = time.time()
        #since this tracking is based on the kalman filter prediction without new detections, every track should be matched with one of the predicted bbox
        detections = [detections[i] for i in u_detection]
        detections_second = [detections_second[i] for i in u_detection_second]
        unmatched_detections = detections + detections_second

        new_thresh=0.5
        while len(u_track)>0 and len(unmatched_detections)>0 and new_thresh >= 0:
            new_thresh -= 0.1
            r_tracked_stracks = [r_tracked_stracks[i] for i in u_track]
            dists = matching.iou_distance(r_tracked_stracks, unmatched_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=new_thresh)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = unmatched_detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            unmatched_detections = [unmatched_detections[i] for i in u_detection] 
      #  print(f"third association time: {(time.time()-third_time)*1000}")

        lost_stracks = []
        for idx in u_track:
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
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
      
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

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