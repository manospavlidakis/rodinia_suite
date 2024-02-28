#!/bin/bash
path=$1
file=$2
selection=$3
#echo "filename: "${file}
#echo "path: "${path}
if [ ${selection} -eq 1 ]
then
  echo "selection: "${selection}
  time taskset -c 4-7 ./nw 512 10 &> ${path}/512_${file}
elif [ ${selection} -eq 2 ]
then
  echo "selection: "${selection}
  time taskset -c 4-7 ./nw 1024 10 &> ${path}/1024_${file}
elif [ ${selection} -eq 3 ]
then
  echo "selection: "${selection}
  time taskset -c 4-7 ./nw 2048 10 &> ${path}/2048_${file}
elif [ ${selection} -eq 4 ]
then
  echo "selection: "${selection}
  time taskset -c 4-7 ./nw 4096 10 &> ${path}/4096_${file}
elif [ ${selection} -eq 5 ]
then
  echo "selection: "${selection}
  time taskset -c 4-7 ./nw 8192 10 &> ${path}/8192_${file}
elif [ ${selection} -eq 6 ]
then
  echo "selection: "${selection}
  time taskset -c 4-7 ./nw 16384 10 &> ${path}/16384_${file}
else
  echo "Not valid choice!"
fi
