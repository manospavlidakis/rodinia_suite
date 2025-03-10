#!/bin/bash
benchmarks=("bfs" "gaussian" "hotspot" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
#benchmarks=("nn" "pathfinder")
total_benchmarks=${#benchmarks[@]}
for str in ${benchmarks[@]}
do
    echo "[ RUN               ] $str"
    cd ${str}
    path=$(pwd)
    ./../find_avg_per_app_break.py $str
    cd - &>/dev/null
done
