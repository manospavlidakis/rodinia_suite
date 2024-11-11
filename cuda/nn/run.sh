#!/bin/bash
path=$1
file=$2
./nn list64k.txt -r 5 -lat 30 -lng 90 -q &> 64k_${file}
./nn list128k.txt -r 5 -lat 30 -lng 90 -q &> 128k_${file}
./nn list256k.txt -r 5 -lat 30 -lng 90 -q &> 256k_${file}
./nn list512k.txt -r 5 -lat 30 -lng 90 -q &> 512k_${file}
./nn list1024k.txt -r 5 -lat 30 -lng 90 -q &> 1024k_${file}
./nn list2048k.txt -r 5 -lat 30 -lng 90 -q &> 2048k_${file}
./nn list4096k.txt -r 5 -lat 30 -lng 90 -q &> 4096k_${file}
