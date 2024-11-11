#!/bin/bash
path=$1
file=$2
#echo "filename: "${file}
#echo "path: "${path}
./gaussian -s 256 &> 256_${file}
./gaussian -s 512 &> 512_${file}
./gaussian -s 1024 &> 1024_${file}
./gaussian -s 2048 &> 2048_${file}
./gaussian -s 4096 &> 4096_${file}
./gaussian -s 8192 &> 8192_${file}
