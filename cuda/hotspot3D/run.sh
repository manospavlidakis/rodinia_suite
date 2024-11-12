#!/bin/bash
path=$1
file=$2
for ((iter=1; iter<=5;iter++))
do
./hotspot3D 512 4 1000 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 output &> 512x4_1000_${iter}_${file}
done
