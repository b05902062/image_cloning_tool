#!bin/bash
if [ $# -eq 3 ]
then
  
	python3 create_poly_int.py $1 .poly_int.poly
	./triangle/triangle -pq .poly_int.poly	
	python3 image_cloning_2.py $1 .poly_int.1.ele .poly_int.1.node .poly_int.1.poly $2 $3

	rm .poly_int.poly .poly_int.1*
else
	
	echo "bash run.sh <source_img> <target_img> <output_img>"


fi;
