import numpy as np
import cv2
import sys
import math

def reduce_size(image):
	img=cv2.imread(image)
	img=cv2.resize(img,(360,480))
	return img

def produce_image():

	trgt=np.zeros((50,100,3),np.uint8)
	for i in range(5):
		for o in range(100):
			trgt.itemset((i+( (o)/4 >0 and int((o)/4) or 0),o,0),255)
			trgt.itemset((-i+( 49-(o)/4 >0 and int(49-(o)/4) or 0),o,0),255)
			

	cv2.namedWindow("window")
	cv2.imshow("window",trgt)
	cv2.waitKey(0)	
	cv2.imwrite("try.jpg",trgt)


def to_trgt(img_x,img_y,s,f):
	trgt_y=math.atan2(img_y,f)*s
	trgt_x=s*math.atan2(img_x,(f**2+img_y**2)**(1/2))
	return trgt_x,trgt_y

def to_img(trgt_x,trgt_y,s,f):
	img_y=f*math.tan(trgt_y/s)
	img_x=math.tan(trgt_x/s)*f*( (1+math.tan(trgt_y/s)**2)**(1/2) )
	return img_x,img_y

def splatting(img,x,y,color,from_x_img,from_y_img,pixel_x,pixel_y):
	x_int=int(math.floor(x))-int(from_x_img)
	y_int=int(math.floor(y))-int(from_y_img)
	total_weight=0
	total_value=0
	for i in range(-1,2):
		for o in range(-1,2):
			#print(x,y,x_int+i,y_int+o)

			if (x_int+i)>=0 and (x_int+i)<pixel_x and (y_int+o)>=0 and (y_int+o)<pixel_y:
				distance=pow(x-from_x_img-x_int,2)+pow(y-from_y_img-y_int,2)
				weight=5*math.exp(-distance)
				total_weight+=weight
				#print(weight)
				total_value+=weight*img.item(x_int+i,y_int+o,color)
	if total_weight==0:
		return 0
	else:
		return total_value/total_weight

def spherical_projection(image_name,f,s):

	img=reduce_size(image_name)#,cv2.IMREAD_COLOR)
	pixel_x ,pixel_y, _ =img.shape
	from_x_img=-math.floor(pixel_x/2)
	from_y_img=-math.floor(pixel_y/2)
	to_x_img=math.floor(pixel_x/2)
	to_y_img=math.floor(pixel_y/2)

	min_x=10**10
	min_y=10**10
	max_x=0
	max_y=0
	trgt_x,trgt_y=to_trgt(from_x_img,from_y_img,s,f)

	if(max_x<trgt_x):
		max_x=trgt_x
	if(max_y<trgt_y):
		max_y=trgt_y
	if(min_x>trgt_x):
		min_x=trgt_x
	if(min_y>trgt_y):
		min_y=trgt_y


	trgt_x,trgt_y=to_trgt(to_x_img,to_y_img,s,f)

	if(max_x<trgt_x):
		max_x=trgt_x
	if(max_y<trgt_y):
		max_y=trgt_y
	if(min_x>trgt_x):
		min_x=trgt_x
	if(min_y>trgt_y):
		min_y=trgt_y
	
	trgt_x,trgt_y=to_trgt(from_x_img,0,s,f)

	if(max_x<trgt_x):
		max_x=trgt_x
	if(max_y<trgt_y):
		max_y=trgt_y
	if(min_x>trgt_x):
		min_x=trgt_x
	if(min_y>trgt_y):
		min_y=trgt_y

	trgt_x,trgt_y=to_trgt(0,to_y_img,s,f)

	if(max_x<trgt_x):
		max_x=trgt_x
	if(max_y<trgt_y):
		max_y=trgt_y
	if(min_x>trgt_x):
		min_x=trgt_x
	if(min_y>trgt_y):
		min_y=trgt_y

	trgt_x,trgt_y=to_trgt(0,from_y_img,s,f)

	if(max_x<trgt_x):
		max_x=trgt_x
	if(max_y<trgt_y):
		max_y=trgt_y
	if(min_x>trgt_x):
		min_x=trgt_x
	if(min_y>trgt_y):
		min_y=trgt_y

	trgt_x,trgt_y=to_trgt(to_x_img,0,s,f)

	if(max_x<trgt_x):
		max_x=trgt_x
	if(max_y<trgt_y):
		max_y=trgt_y
	if(min_x>trgt_x):
		min_x=trgt_x
	if(min_y>trgt_y):
		min_y=trgt_y
	min_x=math.ceil(min_x)
	min_y=math.ceil(min_y)
	max_x=math.ceil(max_x)
	max_y=math.ceil(max_y)

	if max_x==min_x or max_y==min_y:
		print("image_too_small")
		exit()
	print("min_x",min_x,"max_x",max_x,"min_y",min_y,"max_y",max_y)
	trgt_img=np.zeros((max_x-min_x,max_y-min_y,3),dtype=np.uint8)
	for i in range(min_x,max_x):
		for o in range(min_y,max_y):
			img_x,img_y=to_img(i,o,s,f)
			img_x=math.ceil(img_x)-(from_x_img)

			img_y=math.ceil(img_y)-(from_y_img)
			#print([img_x,img_y])
			if img_x<0 or img_y<0 or img_x>=img.shape[0] or img_y>=img.shape[1]:
				continue		
			trgt_img[i-min_x,o-min_y]=img[img_x,img_y]
	cv2.namedWindow("window")
	cv2.imshow("window",trgt_img)
	cv2.waitKey(0)
	cv2.imwrite(image_name[:-4]+"_projected.jpg",trgt_img)
	print(trgt_img.shape)

if __name__=="__main__":

	if len(sys.argv) != 4:
		print("usage: cylindrical_projection <img> <focal_length> <radius>")
		exit(-1)
	
	spherical_projection(sys.argv[1],float(sys.argv[2]),float(sys.argv[3]))
