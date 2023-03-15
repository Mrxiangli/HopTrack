#!/bin/sh

sampling=$1
dis_traj=$2

python track.py --path ../Video_Colab/MOT16_dataset/MOT16-01.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-01.txt;
echo "finish MOT-01"
echo "start MOT-02"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-02.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-02.txt;
echo "finish MOT-02"
echo "start MOT-03"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-03.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling   --save_result > MOT16-03.txt;
echo "finish MOT-03"
echo "start MOT-04"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-04.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-04.txt;
echo "finish MOT-04"
echo "start MOT-05"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-05.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --save_result > MOT16-05.txt;
echo "finish MOT-05"
echo "start MOT-06"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-06.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-06.txt;
echo "finish MOT-06"
echo "start MOT-07"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-07.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-07.txt;
echo "finish MOT-07"
echo "start MOT-08"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-08.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-08.txt;
echo "finish MOT-08"
echo "start MOT-09"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-09.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-09.txt;
echo "finish MOT-09"
echo "start MOT-10"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-10.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-10.txt;
echo "finish MOT-10"
echo "start MOT-11"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-11.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-11.txt;
echo "finish MOT-11"
echo "start MOT-12"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-12.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-12.txt;
echo "finish MOT-12"
echo "start MOT-13"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-13.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-13.txt;
echo "finish MOT-13"
echo "start MOT-14"
python track.py --path ../Video_Colab/MOT16_dataset/MOT16-14.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling  --save_result > MOT16-14.txt;
echo "finish MOT-14"
