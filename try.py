import cv2
import numpy as np

img=np.zeros((300,300,3),dtype='int16')
for i in range(5):
	for o in range(300):
		img[i][o]=(0,0,2**14-1)


cv2.namedWindow("window")
cv2.imshow("window",img)
k=cv2.waitKey(0) 
print(k)
