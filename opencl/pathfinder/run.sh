#!/bin/bash
path=$1
file=$2
selection=$3
if [ ${selection} -eq 1 ]
then 
time taskset -c 4-7 ./pathfinder 1024 1024 2 &> ${path}/1024_${file}
elif [ ${selection} -eq 2 ]
then 
  time taskset -c 4-7 ./pathfinder 2048 2048 2 &> ${path}/2048_${file}
elif [ ${selection} -eq 3 ]
then 
  time taskset -c 4-7 ./pathfinder 4096 4096 2 &> ${path}/4096_${file}
else
  echo "Not valid choice!"
fi
