import cv2
import numpy as np
import sys


def find_edge(mask):
	result = []
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] == 255:
				if i == 0 or i == mask.shape[0]-1:
					result.append([i, j])
				elif j == 0 or j ==	mask.shape[1]-1:
					result.append([i, j])
				else:
					for x in range(3):
						for y in range(3):
							if x == 1 and y == 1:
								continue
							if mask[i-x+1][j-y+1] == 0:
								result.append([i-x+1, j-y+1])
	return result

filename = sys.argv[1]
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
print(img.shape)

b, g, r, a = cv2.split(img)

edge_node = find_edge(a)

with open('test.node', 'w') as f:
	print(len(edge_node), 2, 0, 1, file=f)
	for i in range(len(edge_node)):
		print(i+1, edge_node[i][0], edge_node[i][1], 1, file=f)