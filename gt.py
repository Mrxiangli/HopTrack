import cv2
import sys
from collections import defaultdict

# video_path = '/Users/chen4384/Desktop/Video_Colab/VIRAT_S_010202_06_000784_000873.mp4'
video_path = '/Users/chen4384/Desktop/Video_Colab/VIRAT_S_050301_03_000933_001046.mp4'
# label_path = '/Users/chen4384/Desktop/VIRAT Ground Dataset/annotations/VIRAT_S_010202_06_000784_000873.viratdata.objects.txt'
label_path = 'gt.txt'

label = defaultdict(list)

# with open(label_path) as f:
# 	for line in f:
# 		_, _, frame_id, t, l, w, h, obj = [int(i) for i in line.strip().split()]
# 		label[frame_id].append((t, l, w, h, obj))
with open(label_path) as f:
	for line in f:
		frame_id, obj, t, l, w, h = line.strip().split()
		frame_id = int(frame_id)
		t, l, w, h = [int(float(i)) for i in (t, l, w, h)]
		label[frame_id - 1].append((t, l, w, h, obj))

cap = cv2.VideoCapture(video_path)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"current frame rate: {fps} fps")

save_path = 'gt.mp4'

print(f"video save_path is {save_path}")
vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

ret = False

while ret == False:
	ret, img = cap.read()

frame_id = 0
while True:
	ret_val, img = cap.read()
	if ret_val:

		for x0, y0, w, h, obj in label[frame_id]:

			x1 = x0 + w
			y1 = y0 + h

			color = (255, 0, 0)
			text = str(obj)
			txt_color = (0, 0, 0)
			font = cv2.FONT_HERSHEY_SIMPLEX

			txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
			# print(x0, y0, x1, y1)

			cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

			txt_bk_color = (255, 255, 255)
			cv2.rectangle(
				img,
				(x0, y0 + 1),
				(x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
				txt_bk_color,
				-1
			)
			cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

		frame_id += 1

		vid_writer.write(img)

		ch = cv2.waitKey(1)
		if ch == 27 or ch == ord("q") or ch == ord("Q"):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()

