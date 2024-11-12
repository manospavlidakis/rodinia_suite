#!/bin/bash
path=$1
file=$2
for ((iter=1; iter<=5;iter++))
do
./nn list2048k.txt -r 5 -lat 30 -lng 90 -q &> 2048_${iter}_${file}
done
