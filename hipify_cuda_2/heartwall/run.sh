#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
#echo "Name: ${benchmark_name}"
#echo "filename: "${file}
#echo "path: "${path}
for ((iter=1; iter<=30;iter++))
do
./${benchmark_name} ../../data/heartwall/test.avi 20 &>20_${iter}_${file}
done
../find_avg_per_app.py  ${benchmark_name}
