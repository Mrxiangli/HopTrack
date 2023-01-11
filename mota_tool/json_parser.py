import json

object_list = []

f = open('gt.json')
data = json.load(f)
for index, each_object in enumerate(data[0]["annotations"][0]["result"]):
    #print(each_object)
    object_dict = {}
    object_id = each_object["id"]
    object_dict["object_id"]=object_id
    for idx, each_key in enumerate(each_object["value"]["sequence"]):

        current_frame_label = each_key["frame"]
        cur_x = each_key["x"]
        cur_y = each_key["y"]
        cur_w = each_key["width"]
        cur_h = each_key["height"]

        object_dict[current_frame_label] = (cur_x, cur_y, cur_w, cur_h)
        if idx != len(each_object["value"]["sequence"])-1:
 
            if each_key["enabled"] == True:
                next_frame = each_object["value"]["sequence"][idx+1]
                next_frame_label = each_object["value"]["sequence"][idx+1]["frame"]
                interval = int(next_frame_label) - int(current_frame_label)

                if interval > 1:
            
                    next_x = next_frame["x"]
                    next_y = next_frame["y"]
                    next_w = next_frame["width"]
                    next_h = next_frame["height"]

                    rate_x = ( next_x - cur_x ) / interval
                    rate_y = ( next_y - cur_y ) / interval
                    rate_w = ( next_w - cur_w ) / interval
                    rate_h = ( next_h - cur_h ) / interval

                    for i in range(1,interval):
                        predicted_x = cur_x + i*rate_x
                        predicted_y = cur_y + i*rate_y
                        predicted_w = cur_w + i*rate_w
                        predicted_h = cur_h + i*rate_h
                        predicted_frame_label = current_frame_label+i
                        object_dict[predicted_frame_label] = (predicted_x, predicted_y, predicted_w, predicted_h) 
    object_list.append(object_dict)    
    if index == 43:
        break   

with open("result.json",'w') as fp:
    json.dump(object_list,fp)   