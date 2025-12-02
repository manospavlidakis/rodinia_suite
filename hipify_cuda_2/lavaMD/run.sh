#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
for ((iter=1; iter<=30;iter++))
do
./${benchmark_name} -boxes1d 20 &> 20_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
