#!/bin/bash
############################# Setup ###############################
# Server: Jenna, 1xNUMA node, 16xcores, 0-7 Cores, 8-15 HyperThreads
# GPU Geforce RXT 2080 Ti, Driver 510.54, Cuda 11.6
# Optimal cores for GPU 0-15
# Taskset for Benchamrks 0-7
###################################################################

export smi_pid
function start_smi() {
    cd
    nvidia-smi --query-gpu=timestamp,name,pci.bus,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,clocks.gr,clocks.sm,clocks.mem,clocks.video,power.draw,power.limit --format=csv -lms 100 > ~/smi_results/smi_stats.csv &
    smi_pid=$!
    echo "Start SMI" ${smi_pid}
    sleep 1
    cd - &>/dev/null
}
function stop_smi() {
    echo "Stop SMI" ${smi_pid}
    kill -SIGINT ${smi_pid}
    sleep 10
}

if [ $# -eq 0 ]
then
    echo "Param: 1 for 1xconcurrent, 2 for 2xconcurrent, 4 for 4xconcurrent"
    echo "Param: 1 for MPS, 0 for Native"
    exit
fi

iterations=20
benchmarks=("bfs" "gaussian" "hotspot" "hotspot3D" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
concurrent_instances=$1
MPS=$2

echo "Concurrency: "${concurrent_instances}

server=`echo $HOSTNAME |awk '{split($0,a,"."); print a[1]}'`
#cdate=`date +'%m_%d_%Y'`
#echo ${cdate}
#results=`echo ${cdate}`_${server}_"rush"
results=${server}"_results_rodinia"

mkdir -p ${results}
mkdir -p ${results}_MPS
mkdir -p ${results}_NATIVE

./../../stop_mps.sh
if [ ${MPS} -eq 1 ]
then
	echo "--- START MPS ---"
	./../../start_mps.sh
fi

for str in ${benchmarks[@]}
do
    #Start SMI
    start_smi

    path=${results}/${str}/conc-${concurrent_instances}
    mkdir -p ${path}
    cd ${str}
    for ((i=0; i<${iterations};i++))
    do
        for ((j=1; j<=${concurrent_instances};j++))
        do
            ./execute.sh ${concurrent_instances} ${i} ${str} ${path} 
        done
    done
    cd - &>/dev/null
    #Stop SMI
    stop_smi
    mv ~/smi_results/* ${path}

done
echo "Done Stop Controller"
if [ ${MPS} -eq 1 ]
then
    echo "--- STOP MPS ---"
    cp -rf ${results}/* ${results}_MPS/
    rm -rf ${results}/
    ./../../stop_mps.sh
else
    echo "--- Native ---"
    cp -rf ${results}/* ${results}_NATIVE
    rm -rf ${results}
fi
