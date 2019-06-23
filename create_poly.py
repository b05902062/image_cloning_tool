import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
	print('usage: create_poly.py <input> <output.poly>')
drawing = False # true if mouse is pressed

xi = -1
yi = -1
click = []
finish = False
# mouse callback function
def select_region(event,x,y,flags,param):
	global drawing, xi, yi, click, finish
	
	if finish: 
		return
	
	if event == cv2.EVENT_LBUTTONDOWN:
		if drawing == False:
			drawing = True
		elif abs(x-click[0][0]) < 10 and abs(yi-click[0][1]) < 10:
			finish = True
			return
		click.append((x, y))

	if event == cv2.EVENT_MOUSEMOVE:
		xi = x
		yi = y

filename = sys.argv[1]
output_name = sys.argv[2]

img = cv2.imread(filename)
mask = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
cv2.namedWindow('select')
cv2.setMouseCallback('select',select_region)

alpha = 0.3
while(not finish):
	record = np.zeros(img.shape, dtype=img.dtype)
	for i in range(len(click)-1):
		cv2.line(record, click[i], click[i+1], (0, 0, 255), 3)
	if drawing:
		cv2.line(record, click[-1], (xi, yi), (0, 0, 255))
	output = cv2.addWeighted(record, alpha, img, 1-alpha, 0)
	cv2.imshow('select',output)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27: # esc
		break

cv2.destroyAllWindows()
# print(click)
edge = []
for i in range(len(click)):
	j = (i+1)%len(click)
	diff = np.array([click[j][0]-click[i][0], click[j][1]-click[i][1]], dtype='float64')
	length = np.sqrt(np.sum(np.square(diff)))
	diff = diff / length
	start = np.array(click[i], dtype='float64')
	count = 0
	while(count < length):
		edge.append(start.copy())
		start += diff
		count += 1

with open(output_name, 'w') as f:
	print(len(edge), 2, 0, 1, file=f)
	for i in range(len(edge)):
		print(i+1, edge[i][0], edge[i][1], 1, file = f)
	print(len(edge), 0, file = f)
	for i in range(len(edge)-1):
		print(i+1, i+1, i+2, file = f)
	print(len(edge), len(edge), 1, file = f)
	print(0, file = f)