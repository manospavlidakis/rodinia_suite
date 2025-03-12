#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "../util/cuda/cuda.h"
#include "./kernel_gpu_cuda_2.cu"
#include "./kernel_gpu_cuda_wrapper_2.h"

void kernel_gpu_cuda_wrapper_2(knode *knodes, long knodes_elem, long knodes_mem,

                               int order, long maxheight, int count,

                               long *currKnode, long *offset, long *lastKnode,
                               long *offset_2, int *start, int *end,
                               int *recstart, int *reclength) {

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

  //  printf("# of blocks = %d, # of threads/block = %d (ensure that device can
  //  "
  //         "handle)\n",
  //         numBlocks, threadsPerBlock);
  knode *knodesD;
  cudaMalloc((void **)&knodesD, knodes_mem);
  checkCUDAError("cudaMalloc  recordsD");

  long *currKnodeD;
  cudaMalloc((void **)&currKnodeD, count * sizeof(long));
  checkCUDAError("cudaMalloc  currKnodeD");

  long *offsetD;
  cudaMalloc((void **)&offsetD, count * sizeof(long));
  checkCUDAError("cudaMalloc  offsetD");

  long *lastKnodeD;
  cudaMalloc((void **)&lastKnodeD, count * sizeof(long));
  checkCUDAError("cudaMalloc  lastKnodeD");

  long *offset_2D;
  cudaMalloc((void **)&offset_2D, count * sizeof(long));
  checkCUDAError("cudaMalloc  offset_2D");

  int *startD;
  cudaMalloc((void **)&startD, count * sizeof(int));
  checkCUDAError("cudaMalloc startD");

  int *endD;
  cudaMalloc((void **)&endD, count * sizeof(int));
  checkCUDAError("cudaMalloc endD");

  int *ansDStart;
  cudaMalloc((void **)&ansDStart, count * sizeof(int));
  checkCUDAError("cudaMalloc ansDStart");

  int *ansDLength;
  cudaMalloc((void **)&ansDLength, count * sizeof(int));
  checkCUDAError("cudaMalloc ansDLength");

  cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy memD");

  cudaMemcpy(currKnodeD, currKnode, count * sizeof(long),
             cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy currKnodeD");

  cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy offsetD");

  cudaMemcpy(lastKnodeD, lastKnode, count * sizeof(long),
             cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy lastKnodeD");

  cudaMemcpy(offset_2D, offset_2, count * sizeof(long), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy offset_2D");

  cudaMemcpy(startD, start, count * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy startD");

  cudaMemcpy(endD, end, count * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy endD");

  cudaMemcpy(ansDStart, recstart, count * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy ansDStart");

  cudaMemcpy(ansDLength, reclength, count * sizeof(int),
             cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy ansDLength");

  // [GPU] findRangeK kernel
  findRangeK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem,

                                             currKnodeD, offsetD, lastKnodeD,
                                             offset_2D, startD, endD, ansDStart,
                                             ansDLength);
  cudaDeviceSynchronize();
  checkCUDAError("findRangeK");

  cudaMemcpy(recstart, ansDStart, count * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy ansDStart");

  cudaMemcpy(reclength, ansDLength, count * sizeof(int),
             cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy ansDLength");

  cudaFree(knodesD);
  cudaFree(currKnodeD);
  cudaFree(offsetD);
  cudaFree(lastKnodeD);
  cudaFree(offset_2D);
  cudaFree(startD);
  cudaFree(endD);
  cudaFree(ansDStart);
  cudaFree(ansDLength);
}

#ifdef __cplusplus
}
#endif
