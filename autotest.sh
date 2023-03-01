#!/bin/sh

sampling=$1

python track.py --path ../Video_Colab/MOT16_dataset/MOT16-01.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-01.txt;
echo "finish MOT-01"
sleep 10
echo "start MOT-02"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-02.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-02.txt;
echo "finish MOT-02"
sleep 10
echo "start MOT-03"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-03.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-03.txt;
echo "finish MOT-03"
sleep 10
echo "start MOT-04"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-04.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-04.txt;
echo "finish MOT-04"
sleep 10
echo "start MOT-05"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-05.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-05.txt;
echo "finish MOT-05"
sleep 10
echo "start MOT-06"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-06.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-06.txt;
echo "finish MOT-06"
sleep 10
echo "start MOT-07"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-07.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-07.txt;
echo "finish MOT-07"
sleep 10
echo "start MOT-08"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-08.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-08.txt;
echo "finish MOT-08"
sleep 10
echo "start MOT-09"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-09.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-09.txt;
echo "finish MOT-09"
sleep 10
echo "start MOT-10"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-10.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-10.txt;
echo "finish MOT-10"
sleep 10
echo "start MOT-11"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-11.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-11.txt;
echo "finish MOT-11"
sleep 10
echo "start MOT-12"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-12.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-12.txt;
echo "finish MOT-12"
sleep 10
echo "start MOT-13"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-13.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-13.txt;
echo "finish MOT-13"
sleep 10
echo "start MOT-14"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-14.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-14.txt;
echo "finish MOT-14"

echo "copying files for MOT16 evaluation"
cp MOT16-02.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-02.txt
cp MOT16-04.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-04.txt
cp MOT16-05.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-05.txt
cp MOT16-09.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-09.txt
cp MOT16-10.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-10.txt
cp MOT16-11.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-11.txt
cp MOT16-13.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT16-train/Ours/data/MOT16-13.txt

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

python mot_generator.py
sleep 5

python FPS_calculator.py $1