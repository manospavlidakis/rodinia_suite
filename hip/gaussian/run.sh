#!/bin/bash
path=$1
file=$2
#echo "filename: "${file}

#echo "path: "${path}
for ((iter=1; iter<=5;iter++))
do
./gaussian -s 2048 &> 2048_${iter}_${file}
done
