#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
time taskset -c 4-7 ./hotspot3D 512 8 10 /mnt/vol0/data/hotspot3D/power_512x8 /mnt/vol0/data/hotspot3D/temp_512x8 output &> ${path}/512x8_10_${file}
time taskset -c 4-7 ./hotspot3D 512 8 100 /mnt/vol0/data/hotspot3D/power_512x8 /mnt/vol0/data/hotspot3D/temp_512x8 output &> ${path}/512x8_100_${file}
time taskset -c 4-7 ./hotspot3D 512 8 1000 /mnt/vol0/data/hotspot3D/power_512x8 /mnt/vol0/data/hotspot3D/temp_512x8 output &> ${path}/512x8_1000_${file}

time taskset -c 4-7 ./hotspot3D 512 4 10 /mnt/vol0/data/hotspot3D/power_512x8 /mnt/vol0/data/hotspot3D/temp_512x8 output &> ${path}/512x4_10_${file}
time taskset -c 4-7 ./hotspot3D 512 4 100 /mnt/vol0/data/hotspot3D/power_512x8 /mnt/vol0/data/hotspot3D/temp_512x8 output &> ${path}/512x4_100_${file}
time taskset -c 4-7 ./hotspot3D 512 4 1000 /mnt/vol0/data/hotspot3D/power_512x8 /mnt/vol0/data/hotspot3D/temp_512x8 output &> ${path}/512x4_1000_${file}

