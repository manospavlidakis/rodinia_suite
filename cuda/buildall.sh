#!/bin/bash
SPECTRAL=$3

if [ "$SPECTRAL" = true ]; then
    CUDA_DIR=$1
    SM_VERSION=$2
    # Code that runs if FLAG is true
    echo "Run with Spectral"
else
    CUDA_DIR=/opt/cuda-11.7
    SM_VERSION="86"
    # Code that runs if FLAG is not true
    echo "FLAG is not defined"
fi
echo "CUDA dir: ${CUDA_DIR}"
CUDA_LIB_DIR=$CUDA_DIR/lib64
echo "SM version : ${SM_VERSION}"

GENCODE_FLAGS="-gencode arch=compute_${SM_VERSION},code=sm_${SM_VERSION}"
CXXFLAGS="-std=c++11 -m64 -O3"

#if [ "$d" = "debug" ]; then
CXXFLAGS+=" -DOUTPUT"
#fi

for mf in `find -name 'Makefile'`; do                                                               
    cd `dirname $mf`                                                                                
    make clean                                                                                      
    make -j CUDA_DIR="$CUDA_DIR" GENCODE_FLAGS="$GENCODE_FLAGS" CXXFLAGS="$CXXFLAGS" CUDA_LIB_DIR="$CUDA_LIB_DIR"
    cd -                   
done    
