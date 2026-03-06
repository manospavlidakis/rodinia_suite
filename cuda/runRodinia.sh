#!/bin/bash
ROOT_DIR=$(pwd)
benchmarks=("backprop" "bfs" "b+tree" "cfd" "dwt2d" "gaussian" "heartwall" "hotspot" "hotspot3D" "huffman" "lavaMD" "nn" "nw" "pathfinder")
total_benchmarks=${#benchmarks[@]}
#### Enable those for profiling ####
#RESULT_DIR="${ROOT_DIR}/profiling_results"
#mkdir -p "${RESULT_DIR}"
#export PROFILING=1
#export PATH=/usr/local/cuda/bin:$PATH # enable to find nsys
#rm -rf */*.sqlite
###################################
echo "[===================]"
for str in ${benchmarks[@]}
do
    echo "[ RUN               ] $str"
    cd ${str}
    path=$(pwd)
    # if you want to profile using nsight do PROFILING=1 ./run.sh ...
    ./run.sh ${path} ${str}.csv
    if [[ -f "${str}_nsight.csv" ]]; then
        mv "${str}_nsight.csv" "${RESULT_DIR}/"
    fi

    if [[ -f "nsys_${str}.nsys-rep" ]]; then
        mv "nsys_${str}.nsys-rep" "${RESULT_DIR}/"
    fi
    value=$(grep "Computation" "average.csv" |  awk -F',' '{printf "%.2f", $2}')
    echo "[   AVG GPU time    ] $value ms"
    cd - &>/dev/null
done
echo "[===================] $total_benchmarks apps ran."
