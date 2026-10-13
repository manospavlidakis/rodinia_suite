#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
#echo "Name: ${benchmark_name}"
#echo "filename: "${file}
#echo "path: "${path}
rm -f *_"${file}" average.csv min.csv max.csv median.csv iqr.csv std_dev.csv  # stale logs/stats
fails=0
for ((iter=1; iter<=30;iter++))
do
rm -f rgb.bmp.dwt.r rgb.bmp.dwt.g rgb.bmp.dwt.b; ./${benchmark_name} rgb.bmp -d 1024x1024 -f -5 -l 3 &>53_${iter}_${file} || fails=$((fails+1))
done
../find_avg_per_app.py  ${benchmark_name} || fails=$((fails+1))
exit $(( ${fails:-0} > 0 ))
