#!/bin/bash
function parser(){
  echo "function parser"
  for work in ${workloads[@]}
  do
    echo "workload: " ${work}
    for ((i=0; i<${iterations};i++))
    do
        cat ${work}_${str}_iter-${i}_idd-*_conc-${concurrency}.csv | grep "FPGA time" | \
            awk '{print $3}' >> ../fpga_merged_${work}_${str}_conc-${concurrency}_${i}.out
        cat ${work}_${str}_iter-${i}_idd-*_conc-${concurrency}.csv | grep "Elapsed" | \
            awk '{print $4}' >> ../elapsed_merged_${work}_${str}_conc-${concurrency}_${i}.out
        cat ${work}_${str}_iter-${i}_idd-*_conc-${concurrency}.csv | grep "Computation" | \
            awk '{print $2}' >> ../compute_merged_${work}_${str}_conc-${concurrency}_${i}.out
        cat ${work}_${str}_iter-${i}_idd-*_conc-${concurrency}.csv | grep "Init" | \
            awk '{print $4}' >> ../init_merged_${work}_${str}_conc-${concurrency}_${i}.out
    done
  done
}
if [ $# -eq 0 ]
then
    echo "1st param: Root dir (eg 2_24_2022_jenna/)"
    echo "2th param: 1 for 1xconcurrent, 2 for 2xconcurrent, 4 for 4xconcurrent"
    echo "3th param: Iterations"
    exit
fi

directory=$1
concurrency=$2
iterations=$3


if [ -z ${iterations} ]
then
    echo "Please set iterations!!!"
    exit
fi

if [ -z ${concurrency} ]
then
    echo "Please set concurrency!!!"
    exit
fi
benchmarks=("bfs" "gaussian" "hotspot" "hotspot3D" "lavaMD" "nn" "nw"\
  "particlefilter" "pathfinder")

for str in ${benchmarks[@]}
do
    echo "Parse : " ${str}
    cd ${directory}/${str}/conc-${concurrency}/
    rm -rf ${iterations}_*.out
    if [[ "$str" == "bfs" ]]
    then
      workloads=("1k" "2k" "4k" "8k" "16k" "32k" "64k" "128k" "256k"\
        "512k" "1M")
      parser ${workloads} ${iterations} ${concurrency}
   elif [[ "$str" == "gaussian" ]]
    then
      echo "Gaussian"
      workloads=("256" "512" "1024" "2048")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "hotspot" ]]
    then 
      echo "Hotspot"
      workloads=("512_100k" "512_1M" "512_10M" "512_100M" \
        "1024_100k" "1024_1M" "1024_10M" "1024_100M")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "hotspot3D" ]]
    then 
      echo "Hotspot3D"
      workloads=("512x4_10" "512x4_100" "512x4_1000" \
        "512x8_10" "512x8_100" "512x8_1000")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "lavaMD" ]]
    then 
      echo "lavaMD"
      workloads=("10" "20" "30" "40" "50")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "nn" ]]
    then 
      echo "NN"
      workloads=("64k" "128k" "512k" "1024k" "2048k")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "nw" ]]
    then 
      echo "NW"
      workloads=("512" "1024" "2048" "4096" "8192" "16384")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "particlefilter" ]]
    then 
      echo "Particlefilter"
      workloads=("128_10_10" "128_10_100" "128_10_1000" "128_100_1000"\
        "256_10_10" "256_10_100" "256_10_1000" "256_100_10")
      parser ${workloads} ${iterations} ${concurrency}
    elif [[ "$str" == "pathfinder" ]]
    then 
      echo "Pathfinder"
      workloads=("1024" "2048" "4096" )
      parser ${workloads} ${iterations} ${concurrency}
    fi

    cd -
done

