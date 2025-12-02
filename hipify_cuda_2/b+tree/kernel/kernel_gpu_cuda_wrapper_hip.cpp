
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
#include "./kernel_gpu_cuda_wrapper.h"

__global__ void findK(long height, knode *knodesD, long knodes_elem,
                      record *recordsD, long *currKnodeD, long *offsetD,
                      int *keysD, record *ansD) {

  // private thread IDs
  int thid = threadIdx.x;
  int bid = blockIdx.x;

  // processtree levels
  int i;
  for (i = 0; i < height; i++) {

    // if value is between the two keys
    if ((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] &&
        (knodesD[currKnodeD[bid]].keys[thid + 1] > keysD[bid])) {
      // this conditional statement is inserted to avoid crush due to but in
      // original code "offset[bid]" calculated below that addresses knodes[] in
      // the next iteration goes outside of its bounds cause segmentation fault
      // more specifically, values saved into knodes->indices in the main
      // function are out of bounds of knodes that they address
      if (knodesD[offsetD[bid]].indices[thid] < knodes_elem) {
        offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
      }
    }
    __syncthreads();

    // set for next tree level
    if (thid == 0) {
      currKnodeD[bid] = offsetD[bid];
    }
    __syncthreads();
  }

  // At this point, we have a candidate leaf node which may contain
  // the target record.  Check each key to hopefully find the record
  if (knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]) {
    ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
  }
}
void kernel_gpu_cuda_wrapper(record *records, long records_mem, knode *knodes,
                             long knodes_elem, long knodes_mem, int order, long maxheight, int count,
                             long *currKnode, long *offset, int *keys, record *ans)
{
  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  record *recordsD;
  HIP_CHECK(hipMalloc((void **)&recordsD, records_mem));
  knode *knodesD;
  HIP_CHECK(hipMalloc((void **)&knodesD, knodes_mem));
  long *currKnodeD;
  HIP_CHECK(hipMalloc((void **)&currKnodeD, count * sizeof(long)));
  long *offsetD;
  HIP_CHECK(hipMalloc((void **)&offsetD, count * sizeof(long)));
  int *keysD;
  HIP_CHECK(hipMalloc((void **)&keysD, count * sizeof(int)));
  record *ansD;
  HIP_CHECK(hipMalloc((void **)&ansD, count * sizeof(record)));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(recordsD, records, records_mem, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(currKnodeD, currKnode, count * sizeof(long), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(offsetD, offset, count * sizeof(long), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(keysD, keys, count * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(ansD, ans, count * sizeof(record), hipMemcpyHostToDevice));

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  findK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem, recordsD, currKnodeD, offsetD, keysD, ansD);
  HIP_CHECK(hipDeviceSynchronize());

#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(ans, ansD, count * sizeof(record), hipMemcpyDeviceToHost));

#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipFree(recordsD));
  HIP_CHECK(hipFree(knodesD));
  HIP_CHECK(hipFree(currKnodeD));
  HIP_CHECK(hipFree(offsetD));
  HIP_CHECK(hipFree(keysD));
  HIP_CHECK(hipFree(ansD));

#ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown kernel wrapper 1 #####" << std::endl;
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
