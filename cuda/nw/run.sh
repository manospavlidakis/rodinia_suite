#!/bin/bash
path=$1
file=$2
for ((iter=1; iter<=5;iter++))
do
./nw 8192 10 &> 8192_${iter}_${file}
done
