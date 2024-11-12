#!/bin/bash
benchmarks=("bfs" "gaussian" "hotspot" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
for str in ${benchmarks[@]}
do
    echo "Running ${str}"
    cd ${str}
    path=$(pwd)
    ./run.sh ${path} ${str}.csv
    cd - &>/dev/null
done
