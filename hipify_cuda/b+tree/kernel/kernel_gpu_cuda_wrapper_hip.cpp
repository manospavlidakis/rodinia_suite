
#include <hip/hip_runtime.h>
#include <iostream>
#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "./kernel_gpu_cuda_wrapper.h"

void checkCUDAError1(const char *msg) {
  hipError_t err = hipGetLastError();
  if (hipSuccess != err) {
    // fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    std::cout << "Cuda error " << err << std::endl;
    exit(EXIT_FAILURE);
  }
}
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
                             long knodes_elem, long knodes_mem,

                             int order, long maxheight, int count,

                             long *currKnode, long *offset, int *keys,
                             record *ans) {

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

  // printf("# of blocks = %d, # of threads/block = %d (ensure that device can
  // handle)\n", numBlocks, threadsPerBlock);

  record *recordsD;
  hipMalloc((void **)&recordsD, records_mem);
  checkCUDAError1("hipMalloc  recordsD");

  knode *knodesD;
  hipMalloc((void **)&knodesD, knodes_mem);
  checkCUDAError1("hipMalloc  recordsD");

  long *currKnodeD;
  hipMalloc((void **)&currKnodeD, count * sizeof(long));
  checkCUDAError1("hipMalloc  currKnodeD");

  long *offsetD;
  hipMalloc((void **)&offsetD, count * sizeof(long));
  checkCUDAError1("hipMalloc  offsetD");

  int *keysD;
  hipMalloc((void **)&keysD, count * sizeof(int));
  checkCUDAError1("hipMalloc  keysD");

  record *ansD;
  hipMalloc((void **)&ansD, count * sizeof(record));
  checkCUDAError1("hipMalloc ansD");

  hipMemcpy(recordsD, records, records_mem, hipMemcpyHostToDevice);
  checkCUDAError1("hipMalloc hipMemcpy memD");

  hipMemcpy(knodesD, knodes, knodes_mem, hipMemcpyHostToDevice);
  checkCUDAError1("hipMalloc hipMemcpy memD");

  hipMemcpy(currKnodeD, currKnode, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError1("hipMalloc hipMemcpy currKnodeD");

  hipMemcpy(offsetD, offset, count * sizeof(long), hipMemcpyHostToDevice);
  checkCUDAError1("hipMalloc hipMemcpy offsetD");

  hipMemcpy(keysD, keys, count * sizeof(int), hipMemcpyHostToDevice);
  checkCUDAError1("hipMalloc hipMemcpy keysD");

  hipMemcpy(ansD, ans, count * sizeof(record), hipMemcpyHostToDevice);
  checkCUDAError1("hipMalloc hipMemcpy ansD");

  findK<<<numBlocks, threadsPerBlock>>>(maxheight,

                                        knodesD, knodes_elem,

                                        recordsD,

                                        currKnodeD, offsetD, keysD, ansD);
  hipDeviceSynchronize();
  checkCUDAError1("findK");

  hipMemcpy(ans, ansD, count * sizeof(record), hipMemcpyDeviceToHost);
  checkCUDAError1("hipMemcpy ansD");

  hipFree(recordsD);
  hipFree(knodesD);

  hipFree(currKnodeD);
  hipFree(offsetD);
  hipFree(keysD);
  hipFree(ansD);
}

#ifdef __cplusplus
}
#endif
