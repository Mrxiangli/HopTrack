# HopTrack - A Real-time Multi-Object Tracking Framework for Embedded System

## Abstract
Multi-Object Tracking (MOT) is a challenging computer vision problem due to several factors such as occlusion and varying appearance. State-of-the-art MOT trackers proposed for high-end GPUs run on a frame-by-frame, tracking-by-detection basis, in which a detector is used to detect the objects in each frame,
followed by an association algorithm to link detection across frames and predict the future trajectory. Such algorithms, when directly executed on embedded devices, suffer from a low processing rate (<10 fps), which makes them unsuitable for real-time applications. Previous literature has attempted to run MOT on embedded devices by fusing the detector model with the feature embedding model to reduce the feature extraction latency or combining different trackers to increase tracking accuracy. However, the results are not fully satisfactory as they achieve high processing rates by sacrificing tracking accuracy or vice versa. In this paper, we designed a real-time multi-object tracking system, HopTrack, which specifically targets embedded devices. Instead of extracting deep features from the embedding model, which poses bottlenecks for processing speed, we use appearance features through a discretized static and dynamic matching approach to associate objects across several frames. Moreover, to increase the tracking accuracy, we propose a content-aware dynamic sampling technique that adaptively changes the sampling rate of the detection frame to achieve better tracking accuracy.

## Tracking performance
### Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDsw | FPS* |
|------------|-------|------|------|-------|-------|------|------|------|------|
|MOT16       | 62.91 | 60.83 | 50.35 | 31.9% | 13.4% | 19063 | 46283 | 2278 | 30.61 |
|MOT17       | 63.18 | 60.38 | 50.08 | 30.4% | 14.1% | 49848 | 149985 | 6765 | 30.61 |
|MOT20       | 45.6 | 45.2 | 35.4 | 12.8% | 21.6%  | 40419 | 237887 | 2924 | 13.94 |

* All FPS reported are measured on Nvidia Jetson AGX Xavier 
