#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
time taskset -c 20-23 ./gaussian -s 256 &> ${path}/256_${file}
time taskset -c 20-23 ./gaussian -s 512 &> ${path}/512_${file}
time taskset -c 20-23 ./gaussian -s 1024 &> ${path}/1024_${file}
time taskset -c 20-23 ./gaussian -s 2048 &> ${path}/2048_${file}
#time taskset -c 20-23 ./gaussian -s 4096 &> ${path}/4096_${file}
#time taskset -c 20-23 ./gaussian -s 8192 &> ${path}/8192_${file}
