import numpy as np
import cv2
import sys
import math
import random
def	calculate_eight_parameter(image,image_ref):

	img=cv2.imread(image)
	img_ref=cv2.imread(image_ref)
	x,y,_=img_ref.shape
	x_,y_,_=img.shape
	current_motion=np.array([[1,0,0],[0,1,0],[0,0,1]]))
	

	for z in range(1):
		error=np.zeros([200])
		A=np.zeros([200,8])

		for i in range(x):
			for j in range(y):
				warped_point=current_motion.dot(np.array([x,y,1]))
				warped_point/=warped_point[2]+10**(-10)
				if 


				error[i]=img_ref

	error=



def generate_warp_image(image,image_trgt):
	img=reduce_size(image)
	motion=np.array([[1,0,100],[2,1,0],[0,0,1]])
	warped_img=warp(img,motion)
	cv2.namedWindow("window")
	cv2.imshow("window",warped_img)
	cv2.waitKey(0)
	cv2.imwrite("affine_"+image,warped_img)

def warp(img,motion):
	x,y,_=img.shape
	min_x=10**10
	min_y=10**10
	max_x=0
	max_y=0
	warped_point=motion.dot(np.array([0,0,1]))
	warped_point=warped_point/warped_point[2]

	if(max_x<warped_point[0]):
		max_x=warped_point[0]
	if(max_y<warped_point[1]):
		max_y=warped_point[1]
	if(min_x>warped_point[0]):
		min_x=warped_point[0]
	if(min_y>warped_point[1]):
		min_y=warped_point[1]

	warped_point=motion.dot(np.array([x,y,1]))
	warped_point=warped_point/warped_point[2]

	if(max_x<warped_point[0]):
		max_x=warped_point[0]
	if(max_y<warped_point[1]):
		max_y=warped_point[1]
	if(min_x>warped_point[0]):
		min_x=warped_point[0]
	if(min_y>warped_point[1]):
		min_y=warped_point[1]

	warped_point=motion.dot(np.array([x,0,1]))
	warped_point=warped_point/warped_point[2]

	if(max_x<warped_point[0]):
		max_x=warped_point[0]
	if(max_y<warped_point[1]):
		max_y=warped_point[1]
	if(min_x>warped_point[0]):
		min_x=warped_point[0]
	if(min_y>warped_point[1]):
		min_y=warped_point[1]

	warped_point=motion.dot(np.array([0,y,1]))
	warped_point=warped_point/warped_point[2]

	if(max_x<warped_point[0]):
		max_x=warped_point[0]
	if(max_y<warped_point[1]):
		max_y=warped_point[1]
	if(min_x>warped_point[0]):
		min_x=warped_point[0]
	if(min_y>warped_point[1]):
		min_y=warped_point[1]


	min_x=math.ceil(min_x)
	min_y=math.ceil(min_y)
	max_x=math.ceil(max_x)
	max_y=math.ceil(max_y)

	print("min_x",min_x,"max_x",max_x,"min_y",min_y,"max_y",max_y)
	if max_x == min_x or max_y == min_y :
		print("error:image_too_small")
		exit()
	warped_img=np.zeros((max_x-min_x,max_y-min_y,3),dtype=np.uint8)
	inverse_motion=np.linalg.inv(motion)	
	print("inverse_motion",inverse_motion)
	for i in range(min_x,max_x):
		for j in range(min_y,max_y):
			warped_point=inverse_motion.dot(np.array([i,j,1]))
			#print(warped_point[2])
			warped_point=warped_point/(warped_point[2]+10**(-10))
			x_=math.ceil(warped_point[0])
			y_=math.ceil(warped_point[1])
			#print(i,j,x_,y_)
			if x_< 0 or y_<0 or x_ >= x or y_>= y:
				continue

			warped_img[i-min_x,j-min_y]=splat(x_,y_,img)
	return warped_img

def splat(x,y,img):
	numi=np.zeros([3],dtype=np.uint8)
	deno=0
	for i in range(0,1):
		for j in range(0,1):
			if x+i < 0 or x+i >=img.shape[0] or y+j < 0 or y+j >= img.shape[1]:
				continue
			numi=( 1 / ( (i-0)**2 + (j-0)**2  + 1 ))*img[x+i,y+j]
			deno+=( 1 / ( (i-0)**2 + (j-0)**2  + 1 ))
			#print(x,y,"img",img[x+i,y+j],"warped",numi)
	return numi#deno




def reduce_size(image):
	img=cv2.imread(image)
	img=cv2.resize(img,(360,480))
	return img

if __name__=="__main__":
	generate_warp_image(sys.argv[1],sys.argv[2])
#	calculate_eight_parameter(sys.argv[1],sys.argv[2])
