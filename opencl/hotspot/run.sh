#!/bin/bash
path=$1
file=$2
selection=$3
#echo "filename: "${file}
#echo "path: "${path}
if [ ${selection} -eq 1 ]
then
  echo ${selection}
  time taskset -c 4-7 ./hotspot 1024 4000 100000000 /spare/data/hotspot/temp_1024 /spare/data/hotspot/power_1024 out.txt &> ${path}/1024_100M_${file}
elif [ ${selection} -eq 2 ]
then
  echo ${selection}
  time taskset -c 4-7 ./hotspot 1024 4000 10000000 /spare/data/hotspot/temp_1024 /spare/data/hotspot/power_1024 out.txt &> ${path}/1024_10M_${file}
elif [ ${selection} -eq 3 ]
then
  echo ${selection}
  time taskset -c 4-7 ./hotspot 1024 4000 1000000 /spare/data/hotspot/temp_1024 /spare/data/hotspot/power_1024 out.txt &> ${path}/1024_1M_${file}
elif [ ${selection} -eq 4 ]
then
  echo ${selection}
  time taskset -c 4-7 ./hotspot 1024 4000 100000 /spare/data/hotspot/temp_1024 /spare/data/hotspot/power_1024 out.txt &> ${path}/1024_100k_${file}
elif [ ${selection} -eq 5 ]
then
  echo ${selection}
  time taskset -c 4-7 ./hotspot 512 4000 100000000 /spare/data/hotspot/temp_512 /spare/data/hotspot/power_512 out.txt &> ${path}/512_100M_${file}
elif [ ${selection} -eq 6 ]
then
  echo ${selection}
  time taskset -c 4-7 ./hotspot 512 4000 10000000 /spare/data/hotspot/temp_512 /spare/data/hotspot/power_512 out.txt &> ${path}/512_10M_${file}
elif [ ${selection} -eq 7 ]
then
  time taskset -c 4-7 ./hotspot 512 4000 1000000 /spare/data/hotspot/temp_512 /spare/data/hotspot/power_512 out.txt &> ${path}/512_1M_${file}
elif [ ${selection} -eq 8 ]
then
  time taskset -c 4-7 ./hotspot 512 4000 100000 /spare/data/hotspot/temp_512 /spare/data/hotspot/power_512 out.txt &> ${path}/512_100k_${file}
else
  echo "Not valid choice in Hotspot"
fi
