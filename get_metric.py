import pandas

df = pandas.read_csv('/home/cheng/cheng/TrackEval/data/trackers/mot_challenge/MOT16-train/ours/pedestrian_detailed.csv')

print(df.loc[0, ['MOTA', 'MOTP', 'IDF1', 'CLR_FN', 'CLR_FP', 'IDSW', 'GT_Dets']])
