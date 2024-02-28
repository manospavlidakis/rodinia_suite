#!/bin/bash
path=$1
file=$2
selection=$3
if [ ${selection} -eq 1 ]
then
  time taskset -c 4-7 ./hotspot3D 512 8 10 /spare/data/hotspot3D/power_512x8 /spare/data/hotspot3D/temp_512x8  &> ${path}/512x8_10_${file}
elif [ ${selection} -eq 2 ]
then
  time taskset -c 4-7 ./hotspot3D 512 8 100 /spare/data/hotspot3D/power_512x8 /spare/data/hotspot3D/temp_512x8  &> ${path}/512x8_100_${file}
elif [ ${selection} -eq 3 ]
then
  time taskset -c 4-7 ./hotspot3D 512 8 1000 /spare/data/hotspot3D/power_512x8 /spare/data/hotspot3D/temp_512x8  &> ${path}/512x8_1000_${file}
elif [ ${selection} -eq 4 ]
then
  time taskset -c 4-7 ./hotspot3D 512 4 10 /spare/data/hotspot3D/power_512x8 /spare/data/hotspot3D/temp_512x8  &> ${path}/512x4_10_${file}
elif [ ${selection} -eq 5 ]
then
  time taskset -c 4-7 ./hotspot3D 512 4 100 /spare/data/hotspot3D/power_512x8 /spare/data/hotspot3D/temp_512x8  &> ${path}/512x4_100_${file}
elif [ ${selection} -eq 6 ]
then
  time taskset -c 4-7 ./hotspot3D 512 4 1000 /spare/data/hotspot3D/power_512x8 /spare/data/hotspot3D/temp_512x8  &> ${path}/512x4_1000_${file}
else
  echo "Not valid choice in Hotspot3D!"
fi

