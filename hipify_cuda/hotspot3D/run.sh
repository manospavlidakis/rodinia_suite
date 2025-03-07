#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
for ((iter=1; iter<=30;iter++))
do
./${benchmark_name} 512 4 1000 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 result.txt &> 512x4_1000_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
