#include "./../main.h"
#include "./kernel_gpu_cuda.cu"
#include "./kernel_gpu_cuda_wrapper.h"

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

  cudaMalloc((void **)&d_box_gpu, dim_cpu.box_mem);
  cudaMalloc((void **)&d_rv_gpu, dim_cpu.space_mem);
  cudaMalloc((void **)&d_qv_gpu, dim_cpu.space_mem2);
  cudaMalloc((void **)&d_fv_gpu, dim_cpu.space_mem);

  cudaMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);

  // launch kernel - all boxes
  kernel_gpu_cuda<<<blocks, threads>>>(par_cpu, dim_cpu, d_box_gpu, d_rv_gpu,
                                       d_qv_gpu, d_fv_gpu);
#ifdef DEBUG
  checkCUDAError("Start");
#endif

  cudaMemcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem, cudaMemcpyDeviceToHost);

  cudaFree(d_rv_gpu);
  cudaFree(d_qv_gpu);
  cudaFree(d_fv_gpu);
  cudaFree(d_box_gpu);
}
