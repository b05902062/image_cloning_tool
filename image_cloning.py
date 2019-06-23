import cv2
import numpy as np
import sys
import math
xi = -1
yi = -1
x_mark=-1
y_mark=-1
click = []
finish = False
drawing = False # true if mouse is pressed
mark = False
total_clipped_pixels=0
# mouse callback function
def mousecallback_select_region(event,x,y,flags,param):
	global xi,yi,x_mark,y_mark,click,finish,drawing,mark
	
	if finish: 
		return
	
	if event == cv2.EVENT_LBUTTONDOWN:


		if mark ==False:
			if drawing == False:
				drawing=True
			elif abs(x-click[0][0]) < 10 and abs(y-click[0][1]) < 10:
				mark = True
				drawing = False
				return
			click.append((x, y))

		else:

			finish=True
			x_mark=x
			y_mark=y
			return
	if event == cv2.EVENT_MOUSEMOVE:
		xi = x
		yi = y

def select_region(filename):

	img = cv2.imread(filename)
	print("read image:",filename,"shape:",img.shape)
	cv2.namedWindow('select')
	cv2.setMouseCallback('select',mousecallback_select_region)
	#cv2.imshow('select',img)
	#cv2.waitKey(0)
	while(not mark):
		cpy_img=img.copy()
		for i in range(len(click)-1):
			cv2.line(cpy_img, click[i], click[i+1], (0, 0, 255), 3)
		if drawing:
			cv2.line(cpy_img, click[-1], (xi, yi), (0, 0, 255))
		cv2.imshow('select',cpy_img)
		k = cv2.waitKey(100) & 0xFF
		if k == ord('m'):
			mode = not mode
		elif k == 27: # esc
			break

	#click is the clicks a user made, without appending the first click to the end.
	new_edge, upper_left,lower_right =outline()

	cpy_img=img.copy()
	#user select a point in the selected region.
	for i in range(len(new_edge)):
		for o in range(-1,2):
			for z in range(-1,2):
				cpy_img[new_edge[i][1]+z,new_edge[i][0]+o]=(0,0,255)
	while(not finish):
		cv2.imshow('select',cpy_img)
		cv2.waitKey(100)

	print("x_mark",x_mark,"y_mark",y_mark)	
	cv2.destroyAllWindows()

	print("cliping upper_left",upper_left,"lower_right",lower_right)
	#clip would be a img of size lower_right - upper_left + (1,1)
	clip,mask = clip_for_selected_region(new_edge,img,x_mark,y_mark,upper_left,lower_right)
	print("cliped")
	#cv2.namedWindow('select')
	#cv2.imshow('select',clip)
	#cv2.waitKey(0)

	edge_color=[]
	for i in new_edge:
		edge_color.append(img[i[1],i[0]])

	return clip,mask,new_edge,edge_color,upper_left,lower_right
	
def clip_for_selected_region(edge,img,x_mark,y_mark,upper_left,lower_right):
	mask=np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
	mark_point=(x_mark,y_mark)
	#mask_queue=queue.Queue()
	#mask_queue.put(mark_point)
	mask_queue=set()
	mask_queue.add(mark_point)
	count=0
	while(len(mask_queue)):
		current_point=mask_queue.pop()
		
		count+=1
		
		mask[current_point[1],current_point[0]]=1
		four_neibors=[(current_point[0]+1,current_point[1]+0),(current_point[0]+0,current_point[1]+1),(current_point[0]-1,current_point[1]+0),(current_point[0]+0,current_point[1]-1)]
		for i in four_neibors:
			#print("candidate",i)
			if mask[i[1],i[0]]==0 and (not i in mask_queue) and ( not i in edge) :
				mask_queue.add(i)
				#print("put",i)
		

	for i in range(img.shape[0]):
		for o in range(img.shape[1]):
			if(mask[i][o]==1):
				if( (o,i) in edge):
					print("so wrong")

	global total_clipped_pixels
	total_clipped_pixels=count
	print("clipped",count,"pixels")
	clip=img*mask.reshape(img.shape[0],img.shape[1],1)#broadcast
	return clip[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1],mask[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1]



# print(click)
def edge_valid(edge):
	for i in range(len(edge)):
		diff_x=edge[i][0]-edge[(i+1)%len(edge)][0]
		diff_y=edge[i][1]-edge[(i+1)%len(edge)][1]
		if diff_x > 1 or diff_x < -1 or diff_y > 1 or diff_y < -1:
			print("edge_invalid")
			exit()
	print("edge_valid")

#pixels in the new_edge returned might cross each others.
def outline():
	print("click",click)
	edge = []
	for i in range(len(click)):
		j = (i+1)%len(click)
		diff = np.array([click[j][0]-click[i][0], click[j][1]-click[i][1]], dtype='float64')
		length = np.sqrt(np.sum(np.square(diff)))
		diff = diff / (2*length)
		start = np.array(click[i], dtype='float64')
		edge.append(tuple(start.astype('int32')))
		count = 0
		while(count < length):
			
			start += diff
			edge.append((math.floor(start[0]), math.floor(start[1])))
			count += 1/2
	
	start = np.array(click[0], dtype='int32')
	pre = start
	new_edge=[]
	new_edge.append(tuple(start))
	upper_left=start.copy()
	lower_right=start.copy()
	for i in range(len(edge)):
		if not (np.array_equal(edge[i],pre)):
			new_edge.append(tuple(edge[i]))
			if(upper_left[0]>edge[i][0]):
				upper_left[0]=edge[i][0]
			if(upper_left[1]>edge[i][1]):
				upper_left[1]=edge[i][1]
			if(lower_right[0]<edge[i][0]):
				lower_right[0]=edge[i][0]
			if(lower_right[1]<edge[i][1]):
				lower_right[1]=edge[i][1]
			pre=edge[i]
	#print(new_edge)	
	edge_valid(new_edge)
	return new_edge,upper_left,lower_right

def drag(trgt_file,output_file,MAC_coordinate,clip,mask,edge,edge_color,upper_left,lower_right):
	#clip and mask is np array of size (lower_right[0]-upper_left[0]+1,lower_right[1]-upper_left[1]+1)
	trgt_img=cv2.imread(trgt_file)
	cpy_trgt_img=np.copy(trgt_img)
	cv2.namedWindow("window")
	cv2.setMouseCallback('window',setmousecallback_drag)
	global finish,drawing
	drawing=False
	finish=False
	for i,c_edge in enumerate(edge):
		edge[i]=(c_edge[0]-upper_left[0],c_edge[1]-upper_left[1])#edge becomes coordinate in mask but still in cv form.
	x_offset_np=upper_left[1]-y_mark
	y_offset_np=upper_left[0]-x_mark
	x_len_np=lower_right[1]-upper_left[1]
	y_len_np=lower_right[0]-upper_left[0]
	while(not finish):
		
		y_start_np=xi+y_offset_np
		x_start_np=yi+x_offset_np
		if x_start_np<0 or y_start_np<0 or x_start_np+x_len_np>=trgt_img.shape[0] or y_start_np+y_len_np>=trgt_img.shape[1]:
			cv2.waitKey(100)
			continue
		cloned=clone_clip(MAC_coordinate,clip,trgt_img,mask,edge,edge_color,x_start_np,y_start_np,x_len_np,y_len_np)
		cpy_trgt_img=np.copy(trgt_img)
		#print(cpy_trgt_img[x_start_np:x_start_np+x_len_np+1,y_start_np:y_start_np+y_len_np+1])
		cpy_trgt_img[x_start_np:x_start_np+x_len_np+1,y_start_np:y_start_np+y_len_np+1]+=cloned
		
		if drawing:
			cv2.imwrite(output_file,cpy_trgt_img)
			finish=True
		cv2.imshow("window",cpy_trgt_img)
		cv2.waitKey(100)
	cv2.destroyAllWindows()
	return

def ret_angle(b,a,c):
	#return angle a
	a_b=b-a
	a_c=c-a
	cos_theta=np.sum(a_b*a_c)/(np.sqrt(np.sum(np.square(a_b)))*np.sqrt(np.sum(np.square(a_c))))
	if cos_theta > 1 and cos_theta < 1.01:
		cos_theta=1
	elif cos_theta < -1 and cos_theta > -1.01:
		cos_theta=-1
	return math.acos(cos_theta)


def calculate_MAC_coordinate(mask,edge,upper_left,lower_right):
	"""
	edge is a list of two-tuple
	upper_left and lower_right both are (2,) array
	"""
	count=0
	percent=0
	print("preprocessing_coordinate")
	MAC_coordinate=[]
	for x_np in range(upper_left[1],lower_right[1]+1):
		for y_np in range(upper_left[0],lower_right[0]+1):
			if(mask[x_np-upper_left[1],y_np-upper_left[0]]==1):
				#if((y_np,x_np) in edge):
				#	print("error")
				count+=1
				if((count*100)//total_clipped_pixels > percent):
					percent=(count*100)//total_clipped_pixels
					print(percent,"%")

				pixel_list=[ (math.tan(ret_angle(np.array(edge[(t+1)%len(edge)]),np.array([y_np,x_np]),np.array(c_edge))/2)+math.tan(ret_angle(np.array(edge[(t-1)%len(edge)]),np.array([y_np,x_np]),np.array(c_edge))/2))/int(((y_np-c_edge[0])**2+(x_np-c_edge[1])**2)**(1/2)) for t,c_edge in enumerate(edge)]
				
				
				deno=sum(pixel_list)
				for i,pixel in enumerate(pixel_list):
					pixel_list[i]=pixel/deno
				MAC_coordinate.append(pixel_list)
	#print(MAC_coordinate[0])
	return MAC_coordinate

def clone_clip(MAC_coordinate,clip,trgt_img,mask,edge,edge_color,x_start_np,y_start_np,x_len_np,y_len_np):
	#from x_start_np to x_start_np+x_len_np, last pixel is included.
	cloned=clip.copy()
	diff_color=[]
	#print(trgt_img.shape)
	for i,c_edge in enumerate(edge):
		diff=[]
		if x_start_np+c_edge[1]>= trgt_img.shape[0] or y_start_np+c_edge[0]>=trgt_img.shape[1]:
			print(x_start_np+c_edge[1],trgt_img.shape[0],y_start_np+c_edge[0],trgt_img.shape[1])
		for channel in range(3):
			if trgt_img[x_start_np+c_edge[1],y_start_np+c_edge[0]][channel]-edge_color[i][channel] <245:
				diff.append(trgt_img[x_start_np+c_edge[1],y_start_np+c_edge[0]][channel]-edge_color[i][channel])
			else:
				diff.append(0)
		diff_color.append(np.array(diff))
	

	#print("diff",diff_color)
	#print("clip",list(clip))
	index=0
	for i in range(x_start_np,x_start_np+x_len_np+1):
		for o in range(y_start_np,y_start_np+y_len_np+1):
			if mask[i-x_start_np][o-y_start_np]==1:
				coordinate=MAC_coordinate[index]
				index+=1
				weighted_sum=0
				#coordinate_sum=0
				for d,c in enumerate(coordinate):
					#coordinate_sum+=c
					weighted_sum+=c*diff_color[d]
				cloned[i-x_start_np][o-y_start_np]=clip[i-x_start_np][o-y_start_np]-trgt_img[i][o]+weighted_sum
				#print("weighted",weighted_sum)
				#print(coordinate_sum)
	#print("cloned",list(cloned))
	return cloned

def setmousecallback_drag(event,x,y,flags,param):
	global xi,yi,finish,drawing
	if finish: 
		return
	if event == cv2.EVENT_MOUSEMOVE:
		xi = x
		yi = y

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing=True
		return


"""
with open(output_name, 'w') as f:
	print(len(edge), 2, 0, 1, file=f)
	for i in range(len(edge)):
		print(i+1, edge[i][0], edge[i][1], 1, file = f)
	print(len(edge), 0, file = f)
	for i in range(len(edge)-1):
		print(i+1, i+1, i+2, file = f)
	print(len(edge), len(edge), 1, file = f)
	print(0, file = f)
"""

if __name__=="__main__":
	if len(sys.argv) != 4:
		print('usage: image_cloning.py <source_img> <trgt_img> <output_img>')
	
	source_filename = sys.argv[1]
	trgt_filename = sys.argv[1]
	output_filename = sys.argv[3]
	
	clip,mask,edge,edge_color,upper_left,lower_right=select_region(source_filename)
	MAC_coordinate=calculate_MAC_coordinate(mask,edge,upper_left,lower_right)
	drag(trgt_filename,output_filename,MAC_coordinate,clip,mask,edge,edge_color,upper_left,lower_right)

