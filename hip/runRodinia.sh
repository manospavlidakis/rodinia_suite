#!/bin/bash
############################# Setup ###############################
# Server: Tie0, 4xNUMA node, 64xcores, 0-31 Cores, 32-63 HyperThreads
# GPU AMD
# Optimal cores for GPU 16-23
###################################################################

if [ $# -eq 0 ]
then
    echo "Param: 1 for 1xconcurrent, 2 for 2xconcurrent, 4 for 4xconcurrent"
    exit
fi

iterations=10
benchmarks=("bfs" "gaussian" "hotspot" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
concurrent_instances=$1

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
    for ((i=0; i<${iterations};i++))
    do
        for ((j=1; j<=${concurrent_instances};j++))
        do
            ./execute.sh ${concurrent_instances} ${i} ${str} ${path} 
        done
    done
    cd - &>/dev/null
done
echo "--- Native ---"
cp -rf ${results}/* ${results}_NATIVE
rm -rf ${results}
