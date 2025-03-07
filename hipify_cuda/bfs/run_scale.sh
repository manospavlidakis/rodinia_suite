#!/bin/bash
OUT_DIR=~/third_party_exec/
CUDA_DIR=/home/manos/scale_executables/install/targets/gfx1100
GPU_ARCH=sm_86
source /home/manos/redscale/thirdparty-tests/util/args.sh ${OUT_DIR} ${CUDA_DIR} ${GPU_ARCH}
export LD_LIBRARY_PATH="${CUDA_DIR}/lib"
export SCALE_NONFATAL_EXCEPTIONS=1
export REDSCALE_PROFILE=bfs_profile.txt
export REDSCALE_PROFILE_KERNELS=1
export REDSCALE_PROFILE_MEMOPS=1
./bfs ../../data/bfs/graph65536.txt
unset SCALE_NONFATAL_EXCEPTIONS
unset REDSCALE_PROFILE
