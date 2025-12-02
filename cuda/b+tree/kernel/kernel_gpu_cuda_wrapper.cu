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
double g_btree1_alloc_ms   = 0.0;
double g_btree1_h2d_ms     = 0.0;
double g_btree1_compute_ms = 0.0;
double g_btree1_d2h_ms     = 0.0;
double g_btree1_free_ms    = 0.0;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "../common.h"
#include "../util/cuda/cuda.h"
#include "./kernel_gpu_cuda.cu"
#include "./kernel_gpu_cuda_wrapper.h"

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
  cudaMalloc((void **)&recordsD, records_mem);
  knode *knodesD;
  cudaMalloc((void **)&knodesD, knodes_mem);
  long *currKnodeD;
  cudaMalloc((void **)&currKnodeD, count * sizeof(long));
  long *offsetD;
  cudaMalloc((void **)&offsetD, count * sizeof(long));
  int *keysD;
  cudaMalloc((void **)&keysD, count * sizeof(int));
  record *ansD;
  cudaMalloc((void **)&ansD, count * sizeof(record));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(recordsD, records, records_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(currKnodeD, currKnode, count * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(keysD, keys, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ansD, ans, count * sizeof(record), cudaMemcpyHostToDevice);

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  findK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem, recordsD, currKnodeD, offsetD, keysD, ansD);
  cudaDeviceSynchronize();

#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(ans, ansD, count * sizeof(record), cudaMemcpyDeviceToHost);

#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
  s_b4 = std::chrono::high_resolution_clock::now();
#endif

  cudaFree(recordsD);
  cudaFree(knodesD);
  cudaFree(currKnodeD);
  cudaFree(offsetD);
  cudaFree(keysD);
  cudaFree(ansD);


#ifdef BREAKDOWNS
  e_b4 = std::chrono::high_resolution_clock::now();
  g_btree1_alloc_ms   = std::chrono::duration<double, std::milli>(e_b0 - s_b0).count();
  g_btree1_h2d_ms     = std::chrono::duration<double, std::milli>(e_b2 - s_b2).count();
  g_btree1_compute_ms = std::chrono::duration<double, std::milli>(e_b1 - s_b1).count();
  g_btree1_d2h_ms     = std::chrono::duration<double, std::milli>(e_b3 - s_b3).count();
  g_btree1_free_ms    = std::chrono::duration<double, std::milli>(e_b4 - s_b4).count();
#endif
}

#ifdef __cplusplus
}
#endif
