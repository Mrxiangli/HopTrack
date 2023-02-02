#!/bin/bash

date

rm log.txt

for i in 02 04 05 09 10 11 13
do
  echo "Processing MOT16-$i"
  start=$(date +%s)

  python3 track.py -s video -p datasets/MOT_mp4/MOT16-"$i".mp4 -m yolox_m -c weights/bytetrack_m_mot17.pth.tar \
  --save_result --light > ../TrackEval/data/trackers/mot_challenge/MOT16-train/ours/data/MOT16-"$i".txt

  python3 ../TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 \
  --TRACKERS_TO_EVAL /home/cheng/cheng/TrackEval/data/trackers/mot_challenge/MOT16-train/ours \
  --SEQMAP_FILE /home/cheng/cheng/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-"$i"/seqinfo.ini > /dev/null

  python3 get_metric.py >> log.txt

  echo "Completed in $(($(date +%s) - start)) seconds"

done

python3 mota_calc.py