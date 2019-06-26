import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
	print('usage: select.py <input> <output.png>')
drawing = False # true if mouse is pressed

# mouse callback function
def select_region(event,x,y,flags,param):
	global drawing

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True

	if event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			cv2.circle(mask,(x,y),10,1,-1)
	if event == cv2.EVENT_LBUTTONUP:
		drawing = False

filename = sys.argv[1]
output_name = sys.argv[2]

img = cv2.imread(filename)
mask = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
cv2.namedWindow('select')
cv2.setMouseCallback('select',select_region)

alpha = 0.3
while(1):
	mask_region = np.zeros(img.shape, dtype=img.dtype)
	mask_region[mask == 1] = (0 , 0, 255)
	output = cv2.addWeighted(mask_region, alpha, img, 1-alpha, 0)
	cv2.imshow('select',output)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27: # esc
		break
# print(output.shape)
b, g, r = cv2.split(img)
a = np.zeros(b.shape, dtype=b.dtype)
a[mask == 1] = 255
img_bgra = cv2.merge((b, g, r, a))
cv2.imwrite(output_name, img_bgra)
cv2.destroyAllWindows()