#!/bin/bash
path=$1
file=$2
./pathfinder 1024 1024 2 &> 1024_${file}
./pathfinder 2048 2048 2 &> 2048_${file}
./pathfinder 4096 4096 2 &> 4096_${file}
