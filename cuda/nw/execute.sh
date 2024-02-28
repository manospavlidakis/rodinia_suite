#!/bin/bash
TIMEFORMAT='%3R'
function execute() {
    #echo "!!!!!!Function call!!!!!!"
    #echo "instance id : " $i
    #echo "iteration: " $iteration
    #echo "num_of_kernels: " $num_of_kernels
    #echo "kernel_size: " $kernel_size
    #echo "max concurrent: " $concurrent
    local res=1
    while [ $res != 0 ] ; do
        #echo "iteration: " ${i}
        ./run.sh ../${path}/ ${bench}_iter-${iteration}_idd-${i}_conc-${concurrent}.csv
        res=$?
   done
}
if [ $# -eq 0 ]
then
    echo "1st parameter --> num of Concurrent instances"
    echo "2nd parameter --> iteration"
    echo "3rd parameter --> benchmark"
    echo "4th parameter --> path"
    exit
fi
concurrent=$1
iteration=$2
bench=$3
path=$4
#echo "Concurrent instances: " ${concurrent} ", Iterations: " ${iteration} 
#echo "Benchmark: " ${bench} ", Path: " ${path} 
for ((i=1; i<=${concurrent};i++))
do
    execute $i $iteration $num_of_kernels $benc $path &
done
wait
