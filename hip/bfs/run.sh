#!/bin/bash
path=$1
file=$2
#echo "filename: "${file}
#echo "path: "${path}
for ((iter=1; iter<=5;iter++))
do
./bfs ../../data/bfs/graph65536.txt &>65536_${iter}_${file}
done
