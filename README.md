# HopTrack - A Real-time Multi-Object Tracking Framework for Embedded System

## Abstract
Multi-Object Tracking (MOT) is a challenging computer vision problem due to several factors such as occlusion and varying appearance. State-of-the-art MOT trackers proposed for high-end GPUs run on a frame-by-frame, tracking-by-detection basis, in which a detector is used to detect the objects in each frame,
followed by an association algorithm to link detection across frames and predict the future trajectory. Such algorithms, when directly executed on embedded devices, suffer from a low processing rate (<10 fps), which makes them unsuitable for real-time applications. Previous literature has attempted to run MOT on embedded devices by fusing the detector model with the feature embedding model to reduce the feature extraction latency or combining different trackers to increase tracking accuracy. However, the results are not fully satisfactory as they achieve high processing rates by sacrificing tracking accuracy or vice versa. In this paper, we designed a real-time multi-object tracking system, HopTrack, which specifically targets embedded devices. Instead of extracting deep features from the embedding model, which poses bottlenecks for processing speed, we use appearance features through a discretized static and dynamic matching approach to associate objects across several frames. Moreover, to increase the tracking accuracy, we propose a content-aware dynamic sampling technique that adaptively changes the sampling rate of the detection frame to achieve better tracking accuracy.

## Tracking performance
### HopTrack result on MOT challenge test set
| Dataset    |  MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDsw | FPS* |
|------------|-------|------|------|-------|-------|------|------|------|------|
|MOT16       | 62.91 | 60.83 | 50.35 | 31.9% | 13.4% | 19063 | 46283 | 2278 | 30.61 |
|MOT17       | 63.18 | 60.38 | 50.08 | 30.4% | 14.1% | 49848 | 149985 | 6765 | 30.61 |
|MOT20       | 45.6 | 45.2 | 35.4 | 12.8% | 21.6%  | 40419 | 237887 | 2924 | 13.94 |
* All FPS reported are measured on Nvidia Jetson AGX Xavier 

### State-of-the-arts trackers' performance when process on Nvidia Jetson AGX Xavier (MOT16/17)
| Scheme     |  MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDsw | FPS* |
|------------|-------|------|------|-------|-------|------|------|------|------|
|[ByteTrack](https://github.com/ifzhang/ByteTrack)   | 80.3 | 77.3 | 63.1 | 53.2% | 14.5% | 25491 | 83721 | 2196 |'''diff -10.11 |
|[StrongSort](https://github.com/dyhBUPT/StrongSORT)  | 78.3 | 78.5 | 63.5 | -- | -- | -- | -- | 1446 | 3.2 |
|[DeepSort](https://github.com/nwojke/deep_sort)    | 78.0 | 74.5 | 61.2 | -- | -- | -- | -- | 1821 | 5.9 |
|[JDE*](https://github.com/Zhongdao/Towards-Realtime-MOT) | 73.1 | 68.9 | -- | -- | --  | 6593 | 21788 | 1312 | 9.08 |

JDE is tested on MOT16 only

### Visualization results on MOT challenge test set
<img src="src_files/sample_video/MOT16-01.gif" width="400"/>  <img src="src_files/sample_video/MOT16-03.gif" width="400"/>
<img src="src_files/sample_video/MOT16-09.gif" width="400"/>  <img src="src_files/sample_video/MOT16-13.gif" width="400"/>

## HopTrack Implementation
Since HopTrack is designed to run on embedded devices, a compatible Nvidia jetson Platforms is need in order to replicate our experiments. At the time writing this document, we performed thorough test of HopTrack on NVIDIA Jetson AGX Xavier. The enviroment of the hardware is listed below:
```
Jetpack: 4.6.2 [L4T 32.7.2]
CUDA: 10.2.300
OPENCV: 4.4.0 compiled CUDA: YES
TensorRT: 8.2.1.8 
cuDNN: 8.2.1.32
```

We implemented HopTrack with Python 3.6.9. The rest of the supporting package can be installed with the following command:

```
$pip install -r requirements.txt
```

Since HopTrack is detector independent, it can be adapted with any state-of-the-arts detector that produce the bounding box outputs with the following format 
```
<tx, ty, bx, by, class_id, confidence>
```
We reserved two detector sockets in our implementation, Yolox and Yolov5, their repective weights can be downloaded from [Yolox weighs] and [Yolov5 weights](https://github.com/ultralytics/yolov5).

HopTrack has three different variations - HopDynamo, HopSwift, HopAccurate. To sepcify the variation to execute, the following command can be used:
```
python track.py --path path/to/the/video.mp4 --model yolox_s --ckpt path/to/the/weight.pth.tar --mot --dynamic
```
```
--mot
```
indicates to use the privately trained weights start with "hoptrack_xxx.tar", this only performs tracking on human. For public detectors, remove '--mot' flag and use the corresponding weights.
```
--dynamic 
```
indicates the dynamic sample strategy. using "--swift" or "--accurate" for other two sampling strategies.
```
--dis_traj
```
can be used to enable and disable the trajectory

```
--save_result
```
save the processed video sequence

## HopTrack Evaluation

To replicate the results, it is important to download the original dataset from [MOT16](https://motchallenge.net/data/MOT16/), [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/)

