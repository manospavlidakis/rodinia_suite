
#include <hip/hip_runtime.h>
#include <iostream>
#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "./kernel_gpu_cuda_wrapper_2.h"
void checkCUDAError(const char *msg) {
  hipError_t err = hipGetLastError();
  if (hipSuccess != err) {
    // fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    std::cout << "Cuda error " << err << std::endl;
    exit(EXIT_FAILURE);
  }
}
__global__ void findRangeK(long height,

                           knode *knodesD, long knodes_elem,

                           long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD,
                           int *RecstartD, int *ReclenD) {

  // private thread IDs
  int thid = threadIdx.x;
  int bid = blockIdx.x;

  // ???
  int i;
  for (i = 0; i < height; i++) {

    if ((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) &&
        (knodesD[currKnodeD[bid]].keys[thid + 1] > startD[bid])) {
      // this conditional statement is inserted to avoid crush due to but in
      // original code "offset[bid]" calculated below that later addresses part
      // of knodes goes outside of its bounds cause segmentation fault more
      // specifically, values saved into knodes->indices in the main function
      // are out of bounds of knodes that they address
      if (knodesD[currKnodeD[bid]].indices[thid] < knodes_elem) {
        offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
      }
    }
    if ((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) &&
        (knodesD[lastKnodeD[bid]].keys[thid + 1] > endD[bid])) {
      // this conditional statement is inserted to avoid crush due to but in
      // original code "offset_2[bid]" calculated below that later addresses
      // part of knodes goes outside of its bounds cause segmentation fault more
      // specifically, values saved into knodes->indices in the main function
      // are out of bounds of knodes that they address
      if (knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem) {
        offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
      }
    }
    __syncthreads();

    // set for next tree level
    if (thid == 0) {
      currKnodeD[bid] = offsetD[bid];
      lastKnodeD[bid] = offset_2D[bid];
    }
    __syncthreads();
  }

  // Find the index of the starting record
  if (knodesD[currKnodeD[bid]].keys[thid] == startD[bid]) {
    RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
  }
  __syncthreads();

  // Find the index of the ending record
  if (knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]) {
    ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid] + 1;
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

  //  printf("# of blocks = %d, # of threads/block = %d (ensure that device can
  //  "
  //         "handle)\n",
  //         numBlocks, threadsPerBlock);
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

  hipMemcpy(currKnodeD, currKnode, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy currKnodeD");

  hipMemcpy(offsetD, offset, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy offsetD");

  hipMemcpy(lastKnodeD, lastKnode, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy lastKnodeD");

  hipMemcpy(offset_2D, offset_2, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError("hipMalloc hipMemcpy offset_2D");

  hipMemcpy(startD, start, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy startD");

  hipMemcpy(endD, end, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy endD");

  hipMemcpy(ansDStart, recstart, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError("hipMemcpy ansDStart");

  hipMemcpy(ansDLength, reclength, count * sizeof(int), hipMemcpyHostToDevice);
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

  hipMemcpy(reclength, ansDLength, count * sizeof(int), hipMemcpyDeviceToHost);
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
