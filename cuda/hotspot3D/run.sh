#!/bin/bash
path=$1
file=$2
./hotspot3D 512 4 10 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 output &> 512x4_10_${file}
./hotspot3D 512 4 100 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 output &> 512x4_100_${file}
./hotspot3D 512 4 1000 ../../data/hotspot3D/power_512x4 ../../data/hotspot3D/temp_512x4 output &> 512x4_1000_${file}
