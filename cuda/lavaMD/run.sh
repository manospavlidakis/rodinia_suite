#!/bin/bash
path=$1
file=$2
./lavaMD -boxes1d 10 &> 10_${file}
./lavaMD -boxes1d 20 &> 20_${file}
./lavaMD -boxes1d 30 &> 30_${file}
./lavaMD -boxes1d 40 &> 40_${file}
./lavaMD -boxes1d 50 &> 50_${file}
