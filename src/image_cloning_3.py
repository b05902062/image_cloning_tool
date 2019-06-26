import numpy as np
import cv2
import time
import sys
import trianglemap_model
import math

xi=-1
yi=-1
finish=False
drawing=False
def get_ul_lr(edge_list,nodes):

	upper_left=[10**10,10**10]#in cv form
	lower_right=[-1,-1]
	
	for i in edge_list:
		x=nodes[i][0][0]
		y=nodes[i][0][1]
		if upper_left[0]>x:
			upper_left[0]=x
		if upper_left[1]>y:
			upper_left[1]=y
		if lower_right[0]<x:
			lower_right[0]=x
		if lower_right[1]<y:
			lower_right[1]=y

	return upper_left,lower_right

def setmousecallback_drag(event,x,y,flags,param):
	global xi,yi,finish,drawing
	if finish or drawing: 
		return
	if event == cv2.EVENT_MOUSEMOVE:
		xi = x
		yi = y

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing=True
		return

	
def drag(output_file,mask,p2weight,mesh2boundary,trgt_file,upper_left,lower_right,edge_list,nodes,src_file):
	cursor=[math.floor((upper_left[0]+lower_right[0])/2),math.floor((upper_left[1]+lower_right[1])/2)]
	
	#clip and mask is np array of size (lower_right[0]-upper_left[0]+1,lower_right[1]-upper_left[1]+1)
	trgt_img=cv2.imread(trgt_file)
	src_img=cv2.imread(src_file)
	cv2.namedWindow("window")
	cv2.setMouseCallback('window',setmousecallback_drag)
	cv2.imshow('window',trgt_img)
	cv2.waitKey(10)
	global finish,drawing
	while(not finish):
		x_offset=xi-cursor[0]
		y_offset=yi-cursor[1]
		#print("x_offset",x_offset,"y_offset",y_offset,"upper_left",upper_left,"lower_right",lower_right)
		#print(trgt_img.shape)
		#print("y_start",upper_left[0]+x_offset,"x_start",upper_left[1]+y_offset,"y_end",lower_right[0]+x_offset,"x_end",lower_right[1]+y_offset)
		
		if upper_left[0]+x_offset < 0 or upper_left[1]+y_offset <0 or lower_right[0]+x_offset>=trgt_img.shape[1] or lower_right[1]+y_offset>=trgt_img.shape[0]:
			cv2.imshow('window',trgt_img)
			cv2.waitKey(10)
			continue

		#{id:np.array[b_diff,g_diff,r_diff],...}
		edge_list_color_diff={i: trgt_img[nodes[i][0][1]+y_offset,nodes[i][0][0]+x_offset].astype(int)-src_img[nodes[i][0][1],nodes[i][0][0]].astype(int) for i in edge_list}
		#print("edge_list_color_diff",edge_list_color_diff)	
		mesh_color_diff={}
		for iden,value in mesh2boundary.items():
			color_diff_sum=np.zeros(3)
			for w_id,weight in value.items():
				color_diff_sum+=edge_list_color_diff[w_id]*weight
			mesh_color_diff[iden]= color_diff_sum
		#print("mesh_color_diff",(mesh_color_diff))
		#print("upper_left",upper_left,"lower_right",lower_right)
		pixels_color_diff={}
		for x in range(upper_left[0],lower_right[0]+1):
			for y in range(upper_left[1],lower_right[1]+1):
				nodes_color_sum=np.zeros(3)
				total_weight=0
				#print(x,y,mask[y][x])
				if mask[(y,x)] ==0:
					continue
				#print("p2weight",x,y,p2weight[(x,y)])
				for iden,weight in p2weight[(x,y)].items():
					if iden in edge_list_color_diff:
						nodes_color_sum+=edge_list_color_diff[iden]*weight
					else:
						nodes_color_sum+=mesh_color_diff[iden]*weight
					total_weight+=weight
				pixels_color_diff[(x,y)]=nodes_color_sum
		#print("pixels_color_diff",pixels_color_diff)
		#print("\n\n")
		cloned=np.zeros(src_img.shape)
		for iden,diff in pixels_color_diff.items():
			#print("(x,y)",nodes[iden][0][1],nodes[iden][0][0],diff)
			cloned[(iden[1],iden[0])]=diff
			#print(cloned[(iden[1],iden[0])])
	
		#print("cloned",cloned.shape,list(cloned[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1]))
		#trgt_img_x_start=0+y_offset if 0+y_offset >=0 else 0
		#trgt_img_x_end=src_img.shape[0]+y_offset if src_img.shape[0]+y_offset <= trgt_img.shape[0] else trgt_img.shape[0]
		#trgt_img_y_start=0+x_offset if 0+x_offset>=0 else 0
		#trgt_img_y_end=src_img.shape[1]+x_offset if src_img.shape[1]+x_offset<=trgt_img.shape[1] else trgt_img.shape[1]

		cpy_trgt_img=trgt_img.copy()
		cpy_src_img=src_img.copy().astype(int)
		src_mask= mask==0
		cpy_src_img[src_mask]=0

		
		#make pixels to be cloned in trgt_img 0.
		cpy_trgt_img[upper_left[1]+y_offset:lower_right[1]+y_offset+1,upper_left[0]+x_offset:lower_right[0]+x_offset+1]*=src_mask[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1].reshape(lower_right[1]-upper_left[1]+1,lower_right[0]-upper_left[0]+1,1)#+cloned[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1]
		
		#make pixels to be cloned in trgt_img += cpy_src_img+diff
		#make sure uint8 wouldn't overflow
		cpy_src_img[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1]+=cloned[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1].astype(int)

		overflow_mask= np.zeros(src_img.shape)
		overflow_mask= cpy_src_img< overflow_mask
		cpy_src_img[overflow_mask]=0

		overflow_mask= np.zeros(src_img.shape)+255
		overflow_mask= cpy_src_img> overflow_mask
		cpy_src_img[overflow_mask]=255

		cpy_trgt_img[upper_left[1]+y_offset:lower_right[1]+y_offset+1,upper_left[0]+x_offset:lower_right[0]+x_offset+1]+=cpy_src_img[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1].astype('uint8')
		#cpy_trgt_img[upper_left[1]+y_offset:lower_right[1]+y_offset+1,upper_left[0]+x_offset:lower_right[0]+x_offset+1]+=cloned[upper_left[1]:lower_right[1]+1,upper_left[0]:lower_right[0]+1]

		if drawing == True:
			cv2.imwrite(output_file,cpy_trgt_img)
			finish=True
			drawing=False




		cv2.imshow('window',cpy_trgt_img)
		cv2.waitKey(10)




if __name__=="__main__":
	if len(sys.argv) != 7:
		print('usage: python3 trianglemap_model.py <source_img> <ele> <node> <poly> <trgt_img> <output_img>')
		print('ex. python3 trianglemap_model.py wang.jpg wang.1.ele wang.1.node wang.1.poly mickey.jpg mickey_out.jpg')
		exit(0)
	
	source_filename = sys.argv[1]
	elefile = sys.argv[2]
	nodefile = sys.argv[3]
	polyfile = sys.argv[4]
	trgt_filename = sys.argv[5]
	output_filename = sys.argv[6]
	
	m = trianglemap_model.Mesh_model(source_filename, elefile, nodefile, polyfile)
	print(len(m.edge_list),m.edges)
	upper_left,lower_right=get_ul_lr(m.edge_list,m.nodes)
	drag(output_filename,m.map,m.p2weight,m.mesh2boundary,trgt_filename,upper_left,lower_right,m.edge_list,m.nodes,source_filename)
