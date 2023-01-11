import json

f = open('result.json')
data = json.load(f)
width =1280
height=720

with open('gt.txt', 'a') as fh:

    for each_object in data:
        obid = each_object["object_id"]
        del each_object["object_id"]
        for each_key in each_object.keys():
            frame_id = int(each_key)
            x,y,w,h = each_object[each_key]
            txt = f"{frame_id} {obid} {(x/100)*width} {(y/100)*height} {(w/100)*width} {(h/100)*height}\n"
            fh.write(txt)
