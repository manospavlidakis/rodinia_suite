
#ifdef BREAKDOWNS
#include <chrono>
#endif

#ifdef BREAKDOWNS
static std::chrono::high_resolution_clock::time_point s_b0;
static std::chrono::high_resolution_clock::time_point e_b0;
static std::chrono::high_resolution_clock::time_point s_b1;
static std::chrono::high_resolution_clock::time_point e_b1;
static std::chrono::high_resolution_clock::time_point s_b2;
static std::chrono::high_resolution_clock::time_point e_b2;
static std::chrono::high_resolution_clock::time_point s_b3;
static std::chrono::high_resolution_clock::time_point e_b3;
#endif

#include <hip/hip_runtime.h>
#include <iostream>
#include "../../../common/helper_hip.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "./kernel_gpu_cuda_wrapper_2.h"

__global__ void findRangeK(long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD,
                           long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{
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

void kernel_gpu_cuda_wrapper_2(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count,
                               long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end,
                               int *recstart, int *reclength)
{
  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  knode *knodesD;
  HIP_CHECK(hipMalloc((void **)&knodesD, knodes_mem));
  long *currKnodeD;
  HIP_CHECK(hipMalloc((void **)&currKnodeD, count * sizeof(long)));
  long *offsetD;
  HIP_CHECK(hipMalloc((void **)&offsetD, count * sizeof(long)));
  long *lastKnodeD;
  HIP_CHECK(hipMalloc((void **)&lastKnodeD, count * sizeof(long)));
  long *offset_2D;
  HIP_CHECK(hipMalloc((void **)&offset_2D, count * sizeof(long)));
  int *startD;
  HIP_CHECK(hipMalloc((void **)&startD, count * sizeof(int)));
  int *endD;
  HIP_CHECK(hipMalloc((void **)&endD, count * sizeof(int)));
  int *ansDStart;
  HIP_CHECK(hipMalloc((void **)&ansDStart, count * sizeof(int)));
  int *ansDLength;
  HIP_CHECK(hipMalloc((void **)&ansDLength, count * sizeof(int)));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(currKnodeD, currKnode, count * sizeof(long), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(offsetD, offset, count * sizeof(long), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(lastKnodeD, lastKnode, count * sizeof(long), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(offset_2D, offset_2, count * sizeof(long), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(startD, start, count * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(endD, end, count * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(ansDStart, recstart, count * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(ansDLength, reclength, count * sizeof(int), hipMemcpyHostToDevice));

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  // [GPU] findRangeK kernel
  findRangeK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD,
                                             offset_2D, startD, endD, ansDStart, ansDLength);
  HIP_CHECK(hipDeviceSynchronize());

#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(recstart, ansDStart, count * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(reclength, ansDLength, count * sizeof(int), hipMemcpyDeviceToHost));

#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipFree(knodesD));
  HIP_CHECK(hipFree(currKnodeD));
  HIP_CHECK(hipFree(offsetD));
  HIP_CHECK(hipFree(lastKnodeD));
  HIP_CHECK(hipFree(offset_2D));
  HIP_CHECK(hipFree(startD));
  HIP_CHECK(hipFree(endD));
  HIP_CHECK(hipFree(ansDStart));
  HIP_CHECK(hipFree(ansDLength));

#ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown Computation kernel wrapper 2 #####" << std::endl;
  std::chrono::duration<double, std::milli> allocation = e_b0 - s_b0;
  std::cerr << "Allocation time: " << allocation.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer = e_b2 - s_b2;
  std::cerr << "Transfer time: " << transfer.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> compute = e_b1 - s_b1;
  std::cerr << "Compute time: " << compute.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer2 = e_b3 - s_b3;
  std::cerr << "Transfer Back time: " << transfer2.count() << " ms"
            << std::endl;
  std::cerr << " #################################" << std::endl;
#endif
}

#ifdef __cplusplus
}
#endif
