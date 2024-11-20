#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
#echo "filename: "${file}
#echo "path: "${path}
for ((iter=1; iter<=5;iter++))
do
./${benchmark_name} -s 2048 &> 2048_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
