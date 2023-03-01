import pandas as pd
import xlsxwriter
import sys

if len(sys.argv) != 2:
    assert("Error number of arguments")

if sys.argv[1] == "upper":
    sampling_strategy = 1
if sys.argv[1] == "lower":
    sampling_strategy = 2
if sys.argv[1] == "dynamic":
    sampling_strategy = 0

mot_files = [
    'MOT16-01_time.xlsx',
    'MOT16-02_time.xlsx',
    'MOT16-03_time.xlsx',
    'MOT16-04_time.xlsx',
    'MOT16-05_time.xlsx',
    'MOT16-06_time.xlsx',
    'MOT16-07_time.xlsx',
    'MOT16-08_time.xlsx',
    'MOT16-09_time.xlsx',
    'MOT16-10_time.xlsx',
    'MOT16-11_time.xlsx',
    'MOT16-12_time.xlsx',
    'MOT16-13_time.xlsx',
    'MOT16-14_time.xlsx',
]

workbook = xlsxwriter.Workbook("MOT16_FPS.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "sequence")
worksheet.write(0, 1, "det_fuse")
worksheet.write(0, 2, "post_fuse")
worksheet.write(0, 3, "track_fuse")
worksheet.write(0, 4, "post_track")
worksheet.write(0, 5, "FPS")

for idx, each in enumerate(mot_files):
    df = pd.read_excel(each, engine='openpyxl')
    # drop all NAN
    detect_fuse = df['detect_fuse'].dropna()
    post_detect = df['post_detect'].dropna()
    track_fuse = df['track_fuse'].dropna()
    post_track = df['post_track'].dropna()

    # drop top 5 largest data due to GPU warmup, numba compiling
    detect_fuse.drop(index=detect_fuse.nlargest(5).index, inplace=True)
    post_detect.drop(index=post_detect.nlargest(5).index, inplace=True)
    track_fuse.drop(index=track_fuse.nlargest(5).index, inplace=True)
    post_track.drop(index=post_track.nlargest(5).index, inplace=True)

    avg_det_fuse = detect_fuse.mean()
    avg_post_det = post_detect.mean()
    avg_track = track_fuse.mean()
    avg_post_tk = post_track.mean()

    average_infer = 78.4
    if sampling_strategy == 1:
        if "MOT16-05" in each or "MOT16-06" in each:
            average_rate = 6
        elif "MOT16-13" in each or "MOT16-14" in each:
            average_rate = 8
        else:
            average_rate = 13
    
    elif sampling_strategy == 2: 
        if "MOT16-05" in each or "MOT16-06" in each:
            average_rate = 3
        elif "MOT16-13" in each or "MOT16-14" in each:
            average_rate = 6
        else:
            average_rate = 8

    else:
        if "MOT16-05" in each or "MOT16-06" in each:
            average_rate = 4
        elif "MOT16-13" in each or "MOT16-14" in each:
            average_rate = 7
        else:
            average_rate = 8
    

    det_process = avg_det_fuse + avg_post_det + average_infer
    tk_process = avg_track + avg_post_tk
    one_set_process = det_process + tk_process*(average_rate-1)
    num_set = 1000/one_set_process
    fps = num_set * average_rate

    worksheet.write(idx+1, 0, each)
    worksheet.write(idx+1, 1, avg_det_fuse)
    worksheet.write(idx+1, 2, avg_post_det)
    worksheet.write(idx+1, 3, avg_track)
    worksheet.write(idx+1, 4, avg_post_tk)
    worksheet.write(idx+1, 5, fps)

workbook.close()