#!/bin/bash
path=$1
file=$2
#echo "filename: "${file}
#echo "path: "${path}
./bfs ../../data/bfs/graph4096.txt &>4096_${file}
./bfs ../../data/bfs/graph65536.txt &>65536_${file}
