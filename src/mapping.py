import sys

ele_file = 't.1.ele'
node_file = 't.1.node'
poly_file = 't.1.poly'

ele = {}
first = True
with open(ele_file, 'r') as f:
	for line in f:
		if first:
			first = False
			continue
		index, a, b, c = line.split()
		ele[str(index)] = [str(a), str(b), str(c)]

node = {}
first = True
with open(node_file, 'r') as f:
	for line in f:
		if first:
			first = False
			continue
		index, x, y, is_bon = line.split()
		node[str(index)] = {
			'pos': (int(x), int(y)),
			'is_bon': is_bon == '1'
		}

