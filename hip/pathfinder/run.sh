#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
time taskset -c 20-23 ./pathfinder 1024 1024 2 &> ${path}/1024_${file}
time taskset -c 20-23 ./pathfinder 2048 2048 2 &> ${path}/2048_${file}
time taskset -c 20-23 ./pathfinder 4096 4096 2 &> ${path}/4096_${file}
