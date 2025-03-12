#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "../util/cuda/cuda.h"
#include "./kernel_gpu_cuda.cu"
#include "./kernel_gpu_cuda_wrapper.h"

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
  cudaMalloc((void **)&recordsD, records_mem);
  checkCUDAError("cudaMalloc  recordsD");

  knode *knodesD;
  cudaMalloc((void **)&knodesD, knodes_mem);
  checkCUDAError("cudaMalloc  recordsD");

  long *currKnodeD;
  cudaMalloc((void **)&currKnodeD, count * sizeof(long));
  checkCUDAError("cudaMalloc  currKnodeD");

  long *offsetD;
  cudaMalloc((void **)&offsetD, count * sizeof(long));
  checkCUDAError("cudaMalloc  offsetD");

  int *keysD;
  cudaMalloc((void **)&keysD, count * sizeof(int));
  checkCUDAError("cudaMalloc  keysD");

  record *ansD;
  cudaMalloc((void **)&ansD, count * sizeof(record));
  checkCUDAError("cudaMalloc ansD");

  cudaMemcpy(recordsD, records, records_mem, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy memD");

  cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy memD");

  cudaMemcpy(currKnodeD, currKnode, count * sizeof(long),
             cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy currKnodeD");

  cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy offsetD");

  cudaMemcpy(keysD, keys, count * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy keysD");

  cudaMemcpy(ansD, ans, count * sizeof(record), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMalloc cudaMemcpy ansD");

  findK<<<numBlocks, threadsPerBlock>>>(maxheight,

                                        knodesD, knodes_elem,

                                        recordsD,

                                        currKnodeD, offsetD, keysD, ansD);
  cudaDeviceSynchronize();
  checkCUDAError("findK");

  cudaMemcpy(ans, ansD, count * sizeof(record), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy ansD");

  cudaFree(recordsD);
  cudaFree(knodesD);

  cudaFree(currKnodeD);
  cudaFree(offsetD);
  cudaFree(keysD);
  cudaFree(ansD);
}

#ifdef __cplusplus
}
#endif
