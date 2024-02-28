#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
#time taskset -c 20-23 ./nn list64k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/64k_${file}
#time taskset -c 20-23 ./nn list128k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/128k_${file}
time taskset -c 20-23 ./nn list256k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/256k_${file}
time taskset -c 20-23 ./nn list512k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/512k_${file}
time taskset -c 20-23 ./nn list1024k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/1024k_${file}
time taskset -c 20-23 ./nn list2048k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/2048k_${file}
#time taskset -c 20-23 ./nn list4096k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/4096k_${file}
