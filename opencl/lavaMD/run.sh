#!/bin/bash
path=$1
file=$2
selection=$3
if [ ${selection} -eq 1 ]
then
  time taskset -c 4-7 ./lavaMD -boxes1d 10 &> ${path}/10_${file}
elif [ ${selection} -eq 2 ]
then
  time taskset -c 4-7 ./lavaMD -boxes1d 20 &> ${path}/20_${file}
elif [ ${selection} -eq 3 ]
then
  time taskset -c 4-7 ./lavaMD -boxes1d 30 &> ${path}/30_${file}
elif [ ${selection} -eq 4 ]
then
  time taskset -c 4-7 ./lavaMD -boxes1d 40 &> ${path}/40_${file}
elif [ ${selection} -eq 5 ]
then
  time taskset -c 4-7 ./lavaMD -boxes1d 50 &> ${path}/50_${file}
else
  echo "not valid selection!!!"
fi
