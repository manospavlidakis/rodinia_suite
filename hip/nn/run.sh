#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
for ((iter=1; iter<=30;iter++))
do
./${benchmark_name} list2048k.txt -r 5 -lat 30 -lng 90 -q &> 2048_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
