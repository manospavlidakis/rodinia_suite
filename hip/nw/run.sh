#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
#time taskset -c 20-23 ./nw 512 10 &> ${path}/512_${file}
time taskset -c 20-23 ./nw 1024 10 &> ${path}/1024_${file}
time taskset -c 20-23 ./nw 2048 10 &> ${path}/2048_${file}
time taskset -c 20-23 ./nw 4096 10 &> ${path}/4096_${file}
time taskset -c 20-23 ./nw 8192 10 &> ${path}/8192_${file}
#time taskset -c 20-23 ./nw 16384 10 &> ${path}/16384_${file}
