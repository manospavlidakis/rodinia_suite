#!/bin/bash
path=$1
file=$2
benchmark_name="${file%.csv}"
rm -f *_"${file}" average.csv min.csv max.csv median.csv iqr.csv std_dev.csv  # stale logs/stats
fails=0
for ((iter=1; iter<=30;iter++))
do
rm -f result.txt; ./${benchmark_name}  1024 2 2000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 result.txt &> 1024_2000_${iter}_${file} || fails=$((fails+1))
done
../find_avg_per_app.py  ${benchmark_name} || fails=$((fails+1))
#time taskset -c 4-7 ./hotspot 512 4000 100000000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_100M_${file}
#time taskset -c 4-7 ./hotspot 512 4000 10000000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_10M_${file}
#time taskset -c 4-7 ./hotspot 512 4000 1000000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_1M_${file}
#time taskset -c 4-7 ./hotspot 512 4000 100000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_100k_${file}
exit $(( ${fails:-0} > 0 ))
