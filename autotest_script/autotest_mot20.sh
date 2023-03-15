sampling=$1
dis_traj=$2

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-01.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-01.txt;
echo "finish MOT20-01"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-02.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-02.txt;
echo "finish MOT20-02"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-03.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-03.txt;
echo "finish MOT20-03"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-04.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-04.txt;
echo "finish MOT20-02"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-05.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-05.txt;
echo "finish MOT20-05"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-06.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-06.txt;
echo "finish MOT20-06"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-07.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-07.txt;
echo "finish MOT20-07"
sleep 10

python track.py --path ../Video_Colab/MOT20_dataset/MOT20-08.mp4 --model yolox_s --ckpt ../Video_Colab/yolox_weights/bytetrack_s_mot17.pth.tar --mot --$sampling --$dis_traj --save_result > MOT20-08.txt;
echo "finish MOT20-08"
sleep 10

echo "copying files for MOT20 evaluation"
cp MOT20-01.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/data/MOT20-01.txt
cp MOT20-02.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/data/MOT20-02.txt
cp MOT20-03.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/data/MOT20-03.txt
cp MOT20-05.txt /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/data/MOT20-05.txt

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT20-train/MOT20-01/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/pedestrian_detailed.csv MOT20-01-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT20-train/MOT20-02/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/pedestrian_detailed.csv MOT20-02-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT20-train/MOT20-03/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/pedestrian_detailed.csv MOT20-03-evaluation.csv

python /home/dcsl/Documents/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours --SEQMAP_FILE /home/dcsl/Documents/TrackEval/data/gt/mot_challenge/MOT20-train/MOT20-05/seqinfo.ini 
sleep 5
cp /home/dcsl/Documents/TrackEval/data/trackers/mot_challenge/MOT20-train/Ours/pedestrian_detailed.csv MOT20-05-evaluation.csv

python mot_generator.py
sleep 5

python FPS_calculator.py $1

mkdir HopTrack20_dis_$sampling

mv *.txt HopTrack20_dis_$sampling
mv *.xlsx HopTrack20_dis_$sampling
mv *.csv HopTrack20_dis_$sampling