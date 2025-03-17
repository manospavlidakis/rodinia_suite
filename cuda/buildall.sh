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
CXXFLAGS=" -m64 -O3"

CXXFLAGS+=" -DOUTPUT"

#enable it to breakdown the GPU time
#CXXFLAGS+=" -DBREAKDOWNS

for mf in `find -name 'Makefile'`; do                                                               
dir=$(dirname "$mf")
    
    # Enter the directory
    cd "$dir" || exit  # Exit if cd fails, for safety

    # Extract just the directory base name for comparison
    dir_name=$(basename "$dir")

    # Check if the directory is one of the unsupported ones
    if [ "$dir_name" = "kmeans" ] || [ "$dir_name" = "hybridsort" ]; then
        echo "Kmeans and hybridsort are not supported by SCALE due to texture issues."
    else
        make clean
        make -j CUDA_DIR="$CUDA_DIR" GENCODE_FLAGS="$GENCODE_FLAGS" CXXFLAGS="$CXXFLAGS" CUDA_LIB_DIR="$CUDA_LIB_DIR"
    fi  
    cd - > /dev/null              
done    
