#!/bin/bash
TIMEFORMAT='%3R'
function execute() {
    #echo "!!!!!!Function call!!!!!!"
    ./run.sh ../${path}/ ${bench}_iter-${iteration}_idd-0_conc-${concurrent}.csv ${selection}
}
if [ $# -eq 0 ]
then
    echo "1st parameter --> num of Concurrent instances"
    echo "2nd parameter --> iteration"
    echo "3rd parameter --> benchmark"
    echo "4th parameter --> path"
    echo "5th parameter --> workload"
    exit
fi
concurrent=$1
iteration=$2
bench=$3
path=$4
selection=$5
#echo "Concurrent instances: " ${concurrent} ", Iterations: " ${iteration} 
echo "Benchmark: " ${bench} " , iteration: " ${iteration} " , workload: " ${selection} 
execute ${path} ${iteration} ${benc} ${concurrent} ${selection}
