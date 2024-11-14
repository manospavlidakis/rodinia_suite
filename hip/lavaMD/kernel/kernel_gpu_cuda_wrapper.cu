#include "./../main.h" // (in the main program folder)	needed to recognized input parameters
#include "./../util/device/device.h" // (in library path specified to compiler)	needed by for device functions
#include "./kernel_gpu_cuda.cu" // (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables
#include "./kernel_gpu_cuda_wrapper.h" // (in the current directory)
#include "hip/hip_runtime.h"
#include <iostream>
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
  threads.x = NUMBER_THREADS; // define the number of threads in the block
  threads.y = 1;
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif
  hipMalloc((void **)&d_box_gpu, dim_cpu.box_mem);
  hipMalloc((void **)&d_rv_gpu, dim_cpu.space_mem);
  hipMalloc((void **)&d_qv_gpu, dim_cpu.space_mem2);

  hipMalloc((void **)&d_fv_gpu, dim_cpu.space_mem);

#ifdef BREAKDOWNS
  hipDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif
  hipMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, hipMemcpyHostToDevice);

  hipMemcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem, hipMemcpyHostToDevice);
  hipMemcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2, hipMemcpyHostToDevice);

  hipMemcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem, hipMemcpyHostToDevice);
#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif
  // launch kernel - all boxes
  hipLaunchKernelGGL(kernel_gpu_cuda, dim3(blocks), dim3(threads), 0, 0,
                     par_cpu, dim_cpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu);
#ifdef BREAKDOWNS
  hipDeviceSynchronize();
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif
  hipMemcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem, hipMemcpyDeviceToHost);
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
  hipFree(d_rv_gpu);
  hipFree(d_qv_gpu);
  hipFree(d_fv_gpu);
  hipFree(d_box_gpu);
}
