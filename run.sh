#!/bin/bash

date

python3 track.py -s video -p datasets/MOT_mp4/MOT16-"$1".mp4 -m yolox_s -c weights/bytetrack_s_mot17.pth.tar \
--save_result --light > ../TrackEval/data/trackers/mot_challenge/MOT16-train/ours/data/MOT16-"$1".txt

python3 ../TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 \
--TRACKERS_TO_EVAL /home/cheng/cheng/TrackEval/data/trackers/mot_challenge/MOT16-train/ours \
--SEQMAP_FILE /home/cheng/cheng/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-"$1"/seqinfo.ini