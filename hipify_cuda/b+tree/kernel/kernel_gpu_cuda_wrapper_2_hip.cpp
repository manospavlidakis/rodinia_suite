
#include <hip/hip_runtime.h>
#include <iostream>
#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "./kernel_gpu_cuda_2.cu"
#include "./kernel_gpu_cuda_wrapper_2.h"

void checkCUDAError(const char *msg) {
  hipError_t err = hipGetLastError();
  if (hipSuccess != err) {
    // fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    // printf("Cuda error: %s.\n", msg ));
    std::cout << "Cuda error " << err << std::endl;
    exit(EXIT_FAILURE);
  }
}
void kernel_gpu_cuda_wrapper_2(knode *knodes, long knodes_elem, long knodes_mem,
                               int order, long maxheight, int count,
                               long *currKnode, long *offset, long *lastKnode,
                               long *offset_2, int *start, int *end,
                               int *recstart, int *reclength) {

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;
  knode *knodesD;
  hipMalloc((void **)&knodesD, knodes_mem);
  checkCUDAError("hipMalloc  recordsD");

  long *currKnodeD;
  hipMalloc((void **)&currKnodeD, count * sizeof(long));
  checkCUDAError("hipMalloc  currKnodeD");

  long *offsetD;
  hipMalloc((void **)&offsetD, count * sizeof(long));
  checkCUDAError("hipMalloc  offsetD");

  long *lastKnodeD;
  hipMalloc((void **)&lastKnodeD, count * sizeof(long));
  checkCUDAError("hipMalloc  lastKnodeD");

  long *offset_2D;
  hipMalloc((void **)&offset_2D, count * sizeof(long));
  checkCUDAError("hipMalloc  offset_2D");

  int *startD;
  hipMalloc((void **)&startD, count * sizeof(int));
  checkCUDAError("hipMalloc startD");

  int *endD;
  hipMalloc((void **)&endD, count * sizeof(int));
  checkCUDAError("hipMalloc endD");

  int *ansDStart;
  hipMalloc((void **)&ansDStart, count * sizeof(int));
  checkCUDAError("hipMalloc ansDStart");

  int *ansDLength;
  hipMalloc((void **)&ansDLength, count * sizeof(int));
  checkCUDAError("hipMalloc ansDLength");

  hipMemcpy(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy memD");

  hipMemcpy(currKnodeD, currKnode, count * sizeof(long),
             hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy currKnodeD");

  hipMemcpy(offsetD, offset, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy offsetD");

  hipMemcpy(lastKnodeD, lastKnode, count * sizeof(long),
             hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy lastKnodeD");

  hipMemcpy(offset_2D, offset_2, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy offset_2D");

  hipMemcpy(startD, start, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy startD");

  hipMemcpy(endD, end, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy endD");

  hipMemcpy(ansDStart, recstart, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy ansDStart");

  hipMemcpy(ansDLength, reclength, count * sizeof(int),
             hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy ansDLength");

  // [GPU] findRangeK kernel
  findRangeK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem,

                                             currKnodeD, offsetD, lastKnodeD,
                                             offset_2D, startD, endD, ansDStart,
                                             ansDLength);
  hipDeviceSynchronize();
  checkCUDAError("findRangeK");

  hipMemcpy(recstart, ansDStart, count * sizeof(int), hipMemcpyDeviceToHost);
  checkCUDAError("hipMemcpy ansDStart");

  hipMemcpy(reclength, ansDLength, count * sizeof(int),
             hipMemcpyDeviceToHost);
  checkCUDAError("hipMemcpy ansDLength");

  hipFree(knodesD);
  hipFree(currKnodeD);
  hipFree(offsetD);
  hipFree(lastKnodeD);
  hipFree(offset_2D);
  hipFree(startD);
  hipFree(endD);
  hipFree(ansDStart);
  hipFree(ansDLength);
}

#ifdef __cplusplus
}
#endif
