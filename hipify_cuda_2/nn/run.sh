#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
rm -f *_"${file}" average.csv min.csv max.csv median.csv iqr.csv std_dev.csv  # stale logs/stats
fails=0
for ((iter=1; iter<=30;iter++))
do
rm -f result.txt; ./${benchmark_name} list2048k.txt -r 5 -lat 30 -lng 90 -q &> 2048_${iter}_${file} || fails=$((fails+1))
done
../find_avg_per_app.py  ${benchmark_name} || fails=$((fails+1))
exit $(( ${fails:-0} > 0 ))
