#!/bin/bash
path=$1
file=$2
selection=$3
#echo "filename: "${file}
#echo "path: "${path}
if [ ${selection} -eq 1 ]
then
  time taskset -c 4-7 ./nn list64k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/64k_${file}
elif [ ${selection} -eq 2 ]
then
  time taskset -c 4-7 ./nn list128k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/128k_${file}
elif [ ${selection} -eq 3 ]
then
  time taskset -c 4-7 ./nn list512k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/512k_${file}
elif [ ${selection} -eq 4 ]
then
  time taskset -c 4-7 ./nn list1024k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/1024k_${file}
elif [ ${selection} -eq 5 ]
then
  time taskset -c 4-7 ./nn list2048k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/2048k_${file}
elif [ ${selection} -eq 6 ]
then
  time taskset -c 4-7 ./nn list4096k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/4096k_${file}
elif [ ${selection} -eq 7 ]
then
  echo "Crashes do not run!!!"
###### Not working ######
  #time taskset -c 4-7 ./nn list256k.txt -r 5 -lat 30 -lng 90 -q &> ${path}/256k_${file}
else
  echo "Not valid choice!!"
fi
sleep 2
