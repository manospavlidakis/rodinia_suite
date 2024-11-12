#!/bin/bash
path=$1
file=$2
for ((iter=1; iter<=5;iter++))
do
./pathfinder 4096 4096 2 &> 4096_${iter}_${file}
done
