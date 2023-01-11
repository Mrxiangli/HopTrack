import json

fh=open("gt.txt","r")
lines=sum(1 for lines in fh)
fh.close()
fh=open("gt.txt","r")
frame_dict = {}
for i in range(0,lines):
    data_frame = fh.readline().split(" ")
    if int(data_frame[0]) not in frame_dict.keys():
        frame_dict[int(data_frame[0])] = {}
        frame_dict[int(data_frame[0])][data_frame[1]] = (float(data_frame[2]),float(data_frame[3]), float(data_frame[4]), float(data_frame[5]))
    else:
        frame_dict[int(data_frame[0])][data_frame[1]] = (float(data_frame[2]),float(data_frame[3]), float(data_frame[4]), float(data_frame[5]))

with open("frame_result.json",'w') as fp:
    json.dump(frame_dict,fp)  
