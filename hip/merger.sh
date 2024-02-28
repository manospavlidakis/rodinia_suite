#!/bin/bash
function parser(){
  echo "concurency" ${concurrency}
  echo "function parser"
  for work in ${workloads[@]}
  do
    echo "workload: " ${work}
    cat AVG_elapsed_${work}_concurrency_${concurrency}_${str}.final >> ../${str}_total.merged 
    cat AVG_init_${work}_concurrency_${concurrency}_${str}.final >> ../${str}_init.merged 
    cat AVG_compute_${work}_concurrency_${concurrency}_${str}.final >> ../${str}_compute.merged 
    cat AVG_warmup_${work}_concurrency_${concurrency}_${str}.final >> ../${str}_warmup.merged
  done

  tr "\n" "," < ../${str}_init.merged &> ../${str}_init.trans
  tr "\n" "," < ../${str}_compute.merged &> ../${str}_compute.trans
  tr "\n" "," < ../${str}_total.merged &> ../${str}_total.trans
  tr "\n" "," < ../${str}_warmup.merged &> ../${str}_warmup.trans
  paste -d "\n" ../${str}_init.trans ../${str}_compute.trans ../${str}_total.trans ../${str}_warmup.trans &> ../../${str}_all.csv
  rm -rf ../${str}_*.trans 
}
if [ $# -eq 0 ]
then
    echo "1st param: Root dir (eg 2_24_2022_jenna/)"
    echo "2th param: 1 for 1xconcurrent, 2 for 2xconcurrent, 4 for 4xconcurrent"
    exit
fi

directory=$1
concurrency=$2

echo "concurency" ${concurrency}
if [ -z ${concurrency} ]
then
    echo "Please set concurrency!!!"
    exit
fi
## Convert columns to rows
#tr "\n" "," < lavaMD_init.merged
benchmarks=("bfs" "gaussian" "hotspot" "lavaMD" "nn" "nw"\
  "particlefilter" "pathfinder")
for str in ${benchmarks[@]}
do
    echo "Parse : " ${str}
    cd ${directory}/${str}/
    if [[ "$str" == "bfs" ]]
    then
      workloads=("1k" "2k" "4k" "8k" "16k" "32k" "64k" "128k" "256k"\
        "512k" "1M" "2M" "4M")
      parser ${workloads} ${concurrency} ${str}
   elif [[ "$str" == "gaussian" ]]
    then
      echo "Gaussian"
      workloads=("256" "512" "1024" "2048")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "hotspot" ]]
    then 
      echo "Hotspot"
      workloads=("512_10M" "512_100M" "1024_1M" "1024_10M")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "hotspot3D" ]]
    then 
      echo "Hotspot3D"
      workloads=("512x4_10" "512x4_100" "512x4_1000" \
        "512x8_10" "512x8_100" "512x8_1000")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "lavaMD" ]]
    then 
      echo "lavaMD"
      workloads=("20" "30" "40")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "nn" ]]
    then 
      echo "NN"
      workloads=("256k" "512k" "1024k" "2048k")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "nw" ]]
    then 
      echo "NW"
      workloads=("1024" "2048" "4096" "8192")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "particlefilter" ]]
    then 
      echo "Particlefilter"
      workloads=("128_10_1000" "128_100_1000" "256_10_10" "256_10_100" "256_10_1000" "256_100_10")
      parser ${workloads} ${concurrency} ${str}
    elif [[ "$str" == "pathfinder" ]]
    then 
      echo "Pathfinder"
      workloads=("1024" "2048" "4096" )
      parser ${workloads} ${concurrency} ${str}
    fi

    cd -
done

