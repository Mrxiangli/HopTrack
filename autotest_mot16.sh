#!/bin/sh
model=$1
weight=$2
inputdirectory=$3
sampling=$4
dis_traj=$5

for file in "$inputdirectory"/*;do
    echo "start processing $file"
    file_name=$(basename "$file")
    new_filename=${file_name%.mp4}.txt
    python track.py --path $file --model $model --ckpt $weight --mot --$sampling --save_result > $new_filename;
    echo "finished $file"
    new_path=/home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/$new_filename
    cp $new_filename $new_path
    sleep 10
done


python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-02/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-02-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-04/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-04-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-05/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-05-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-09/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-09-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-10/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-10-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-11/seqinfo.ini
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-11-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-13/seqinfo.ini
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/pedestrian_detailed.csv MOT16-13-evaluation.csv

python mot_summary_generator.py
sleep 5
