#!/bin/bash

export CUDA_HOME="/opt/cuda-11.7/"
export LIBNVVM_HOME=${CUDA_HOME}/nvvm
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PATH=${CUDA_HOME}/bin/cuobjdump:${PATH}

benchmarks=("bfs" "gaussian" "hotspot" "hotspot3D" "lavaMD" "nn" "nw" "particlefilter" "pathfinder")
for str in ${benchmarks[@]}
do
    echo "Running ${str}"
    cd ${str}
    path=$(pwd)
    ./run.sh ${path} ${str}.csv
    cd - &>/dev/null
done
