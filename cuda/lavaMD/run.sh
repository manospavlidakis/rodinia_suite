#!/bin/bash
path=$1
file=$2
echo "filename: "${file}
echo "path: "${path}
time taskset -c 4-7 ./lavaMD -boxes1d 10 &> ${path}/10_${file}
time taskset -c 4-7 ./lavaMD -boxes1d 20 &> ${path}/20_${file}
time taskset -c 4-7 ./lavaMD -boxes1d 30 &> ${path}/30_${file}
time taskset -c 4-7 ./lavaMD -boxes1d 40 &> ${path}/40_${file}
time taskset -c 4-7 ./lavaMD -boxes1d 50 &> ${path}/50_${file}
