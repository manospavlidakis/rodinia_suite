#!/bin/bash
path=$1
file=$2
./nn list4.txt -r 5 -lat 30 -lng 90 -q &> 64k_${file}
