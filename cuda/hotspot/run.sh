#!/bin/bash
path=$1
file=$2
./hotspot 1024 4000 100000000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output &> 1024_100M_${file}
./hotspot 1024 4000 10000000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output &> 1024_10M_${file}
./hotspot 1024 4000 1000000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output &> 1024_1M_${file}
./hotspot 1024 4000 100000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output &> 1024_100k_${file}

#time taskset -c 4-7 ./hotspot 512 4000 100000000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_100M_${file}
#time taskset -c 4-7 ./hotspot 512 4000 10000000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_10M_${file}
#time taskset -c 4-7 ./hotspot 512 4000 1000000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_1M_${file}
#time taskset -c 4-7 ./hotspot 512 4000 100000 ../..//data/hotspot/temp_512 ../..//data/hotspot/power_512 output &> ${path}/512_100k_${file}
