#!/bin/bash
benchmarks=("bfs" "gaussian" "hotspot" "hotspot3D" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
#benchmarks=("nn" "pathfinder")
total_benchmarks=${#benchmarks[@]}
echo "[===================]"
for str in ${benchmarks[@]}
do
    echo "[ RUN               ] $str"
    cd ${str}
    path=$(pwd)
    ./run.sh ${path} ${str}.csv
    value=$(grep "Computation" "average.csv" |  awk -F',' '{printf "%.2f", $2}')
    echo "[   AVG GPU time    ] $value ms"
    cd - &>/dev/null
done
echo "[===================] $total_benchmarks apps ran."
