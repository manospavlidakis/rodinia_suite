#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
for ((iter=1; iter<=30;iter++))
do
./${benchmark_name} 4096 4096 2 &> 4096_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
