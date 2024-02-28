#!/bin/bash
############################# Setup ###############################
# Server: Jenna, 1xNUMA node, 16xcores, 0-7 Cores, 8-15 HyperThreads
# GPU Geforce RXT 2080 Ti, Driver 510.54, Cuda 11.6
# Optimal cores for GPU 0-15
# Taskset for Benchamrks 0-7
###################################################################

if [ $# -eq 0 ]
then
    echo "Param: 1 for 1xconcurrent, 2 for 2xconcurrent, 4 for 4xconcurrent"
    exit
fi

iterations=10
benchmarks=("bfs" "gaussian" "hotspot" "hotspot3D" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
#benchmarks=("particlefilter")
concurrent_instances=$1
workloads_bfs=12
workloads_gaussian=6
workloads_hotspot=8
workloads_hotspot3D=6
workloads_lavaMD=5
workloads_nw=6
workloads_nn=7
workloads_particlefilter=12
workloads_pathfinder=3

echo "Concurrency: "${concurrent_instances}

server=`echo $HOSTNAME |awk '{split($0,a,"."); print a[1]}'`
#cdate=`date +'%m_%d_%Y'`
#echo ${cdate}
#results=`echo ${cdate}`_${server}_"rush"
results=${server}"_results_rodinia"

mkdir -p ${results}
mkdir -p ${results}_NATIVE

for str in ${benchmarks[@]}
do
    path=${results}/${str}/conc-${concurrent_instances}
    mkdir -p ${path}
    cd ${str}
    work_array=workloads_${str}
    #echo "work array: " ${work_array}
    for ((j=1; j<=${work_array}; j++))
    do
      #echo "j: "${j}
      for ((i=1; i<${iterations};i++))
      do
        echo "Executing bench : " ${str} "iteration: " ${i} "workload: " ${j}
        #echo "i: " ${i}
        ./execute.sh ${concurrent_instances} ${i} ${str} ${path} ${j} 
      done
    done
    cd - &>/dev/null
done
echo "Done Stop Controller"
echo "--- Native ---"
cp -rf ${results}/* ${results}_NATIVE
rm -rf ${results}
