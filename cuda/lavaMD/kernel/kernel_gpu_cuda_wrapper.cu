#include <chrono>
#include <cuda.h>
#include <iostream>
#include "./../main.h"
#include "./kernel_gpu_cuda.cu"
#include "./kernel_gpu_cuda_wrapper.h"
#define BREAKDOWNS
#ifdef BREAKDOWNS
std::chrono::high_resolution_clock::time_point s_b0;
std::chrono::high_resolution_clock::time_point e_b0;
std::chrono::high_resolution_clock::time_point s_b1;
std::chrono::high_resolution_clock::time_point e_b1;
std::chrono::high_resolution_clock::time_point s_b2;
std::chrono::high_resolution_clock::time_point e_b2;
std::chrono::high_resolution_clock::time_point s_b3;
std::chrono::high_resolution_clock::time_point e_b3;
  
#endif


void kernel_gpu_cuda_wrapper(par_str par_cpu, dim_str dim_cpu, box_str *box_cpu,
                             FOUR_VECTOR *rv_cpu, fp *qv_cpu,
                             FOUR_VECTOR *fv_cpu) {
  box_str *d_box_gpu;
  FOUR_VECTOR *d_rv_gpu;
  fp *d_qv_gpu;
  FOUR_VECTOR *d_fv_gpu;

  dim3 threads;
  dim3 blocks;

  blocks.x = dim_cpu.number_boxes;
  blocks.y = 1;
  threads.x = NUMBER_THREADS;
  threads.y = 1;
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif
  cudaMalloc((void **)&d_box_gpu, dim_cpu.box_mem);
  cudaMalloc((void **)&d_rv_gpu, dim_cpu.space_mem);
  cudaMalloc((void **)&d_qv_gpu, dim_cpu.space_mem2);
  cudaMalloc((void **)&d_fv_gpu, dim_cpu.space_mem);
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif
  // launch kernel - all boxes
  kernel_gpu_cuda<<<blocks, threads>>>(par_cpu, dim_cpu, d_box_gpu, d_rv_gpu,
                                       d_qv_gpu, d_fv_gpu);
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif
#ifdef DEBUG
  checkCUDAError("Start");
#endif

  cudaMemcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem, cudaMemcpyDeviceToHost);
#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif
  #ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown Computation #####" << std::endl;
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
  cudaFree(d_rv_gpu);
  cudaFree(d_qv_gpu);
  cudaFree(d_fv_gpu);
  cudaFree(d_box_gpu);
}
