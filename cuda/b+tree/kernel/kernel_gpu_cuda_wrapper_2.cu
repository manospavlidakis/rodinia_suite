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
static std::chrono::high_resolution_clock::time_point s_b4;
static std::chrono::high_resolution_clock::time_point e_b4;
extern "C" {
double g_btree2_alloc_ms   = 0.0;
double g_btree2_h2d_ms     = 0.0;
double g_btree2_compute_ms = 0.0;
double g_btree2_d2h_ms     = 0.0;
double g_btree2_free_ms    = 0.0;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "../util/cuda/cuda.h"
#include "./kernel_gpu_cuda_2.cu"
#include "./kernel_gpu_cuda_wrapper_2.h"

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
  cudaMalloc((void **)&knodesD, knodes_mem);
  long *currKnodeD;
  cudaMalloc((void **)&currKnodeD, count * sizeof(long));
  long *offsetD;
  cudaMalloc((void **)&offsetD, count * sizeof(long));
  long *lastKnodeD;
  cudaMalloc((void **)&lastKnodeD, count * sizeof(long));
  long *offset_2D;
  cudaMalloc((void **)&offset_2D, count * sizeof(long));
  int *startD;
  cudaMalloc((void **)&startD, count * sizeof(int));
  int *endD;
  cudaMalloc((void **)&endD, count * sizeof(int));
  int *ansDStart;
  cudaMalloc((void **)&ansDStart, count * sizeof(int));
  int *ansDLength;
  cudaMalloc((void **)&ansDLength, count * sizeof(int));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(currKnodeD, currKnode, count * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(lastKnodeD, lastKnode, count * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(offset_2D, offset_2, count * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(startD, start, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(endD, end, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ansDStart, recstart, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ansDLength, reclength, count * sizeof(int),  cudaMemcpyHostToDevice);

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  // [GPU] findRangeK kernel
  findRangeK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD,
                                             offset_2D, startD, endD, ansDStart, ansDLength);
  cudaDeviceSynchronize();

#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(recstart, ansDStart, count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(reclength, ansDLength, count * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
  s_b4 = std::chrono::high_resolution_clock::now();
#endif

  cudaFree(knodesD);
  cudaFree(currKnodeD);
  cudaFree(offsetD);
  cudaFree(lastKnodeD);
  cudaFree(offset_2D);
  cudaFree(startD);
  cudaFree(endD);
  cudaFree(ansDStart);
  cudaFree(ansDLength);

#ifdef BREAKDOWNS
  e_b4 = std::chrono::high_resolution_clock::now();
  g_btree2_alloc_ms   = std::chrono::duration<double, std::milli>(e_b0 - s_b0).count();
  g_btree2_h2d_ms     = std::chrono::duration<double, std::milli>(e_b2 - s_b2).count();
  g_btree2_compute_ms = std::chrono::duration<double, std::milli>(e_b1 - s_b1).count();
  g_btree2_d2h_ms     = std::chrono::duration<double, std::milli>(e_b3 - s_b3).count();
  g_btree2_free_ms    = std::chrono::duration<double, std::milli>(e_b4 - s_b4).count();
#endif
}

#ifdef __cplusplus
}
#endif
