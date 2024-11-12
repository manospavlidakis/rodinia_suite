#!/bin/bash
path=$1
file=$2
for ((iter=1; iter<=5;iter++))
do
./lavaMD -boxes1d 20 &> 20_${iter}_${file}
done
