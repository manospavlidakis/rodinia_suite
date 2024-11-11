#!/bin/bash
#CUDA_DIR=/opt/cuda
echo "Hereeeeeeeeeeeeeeeee!!!"
CUDA_DIR=$1
CUDA_LIB_DIR=$(CUDA_DIR)/lib64
SM_VERSION=$2
numeric_part="${SM_VERSION#*_}"
echo "$numeric_part"

GENCODE_FLAGS="-gencode arch=compute_${SM_VERSION},code=sm_${SM_VERSION}"
CXXFLAGS="-std=c++11 -m64 -O3"

for mf in `find -name 'Makefile'`; do                                                               
    cd `dirname $mf`                                                                                
    make clean                                                                                      
    make -j CUDA_DIR="$CUDA_DIR" GENCODE_FLAGS="$GENCODE_FLAGS" CXXFLAGS="$CXXFLAGS" CUDA_LIB_DIR="$CUDA_LIB_DIR"
    cd -                   
done    
