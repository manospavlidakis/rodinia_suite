/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA. Main file.
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the MIT License. Read the full licence:
 * http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home
 * page in your work.
 *
 */

#include "comparison_helpers.h"
#include "cpuencode.h"
#include "cuda_helpers.h"
#include "load_data.h"
#include "pack_kernels_hip.cpp"
#include "print_helpers.h"
#include "scan_hip.cpp"
#include "stats_logger.h"
#include "stdafx.h"
#include "vlc_kernel_sm64huff_hip.cpp"
#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include <sys/time.h>
#define WARMUP
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks = 1);

extern "C" void cpu_vlc_encode(unsigned int *indata, unsigned int num_elements,
                               unsigned int *outdata, unsigned int *outsize,
                               unsigned int *codewords,
                               unsigned int *codewordlens);

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  if (!InitCUDA()) {
    return 0;
  }
  unsigned int num_block_threads = 256;
  // std::cout << "Argc: " << argc << std::endl;
  if (argc > 1) {
    for (int i = 1; i < argc; i++)
      runVLCTest(argv[i], num_block_threads);
  } else {
    runVLCTest(NULL, num_block_threads, 1024);
  }
  CUDA_SAFE_CALL(hipDeviceReset());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli = end - start;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
  return 0;
}

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks) {
  // printf("CUDA! Starting VLC Tests!\n");
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  double *warm;
  hipMalloc((void **)&warm, sizeof(double) * 100000);
  hipStream_t stream;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipFree(warm);
  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  unsigned int
      num_elements;      // uint num_elements = num_blocks * num_block_threads;
  unsigned int mem_size; // uint mem_size = num_elements * sizeof(int);
  unsigned int symbol_type_size = sizeof(int);
  //////// LOAD DATA ///////////////
  double H; // entropy
  initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size,
             symbol_type_size);
  // printf("Parameters: num_elements: %d, num_blocks: %d, num_block_threads: "
  //        "%d\n----------------------------\n",
  //        num_elements, num_blocks, num_block_threads);
  ////////LOAD DATA ///////////////
  uint *sourceData = (uint *)malloc(mem_size);
  uint *destData = (uint *)malloc(mem_size);
  uint *crefData = (uint *)malloc(mem_size);

  uint *codewords = (uint *)malloc(NUM_SYMBOLS * symbol_type_size);
  uint *codewordlens = (uint *)malloc(NUM_SYMBOLS * symbol_type_size);

  uint *cw32 = (uint *)malloc(mem_size);
  uint *cw32len = (uint *)malloc(mem_size);
  uint *cw32idx = (uint *)malloc(mem_size);

  uint *cindex2 = (uint *)malloc(num_blocks * sizeof(int));

  memset(sourceData, 0, mem_size);
  memset(destData, 0, mem_size);
  memset(crefData, 0, mem_size);
  memset(cw32, 0, mem_size);
  memset(cw32len, 0, mem_size);
  memset(cw32idx, 0, mem_size);
  memset(codewords, 0, NUM_SYMBOLS * symbol_type_size);
  memset(codewordlens, 0, NUM_SYMBOLS * symbol_type_size);
  memset(cindex2, 0, num_blocks * sizeof(int));
  //////// LOAD DATA ///////////////
  loadData(file_name, sourceData, codewords, codewordlens, num_elements,
           mem_size, H);

  //////// LOAD DATA ///////////////

  unsigned int *d_sourceData, *d_destData, *d_destDataPacked;
  unsigned int *d_codewords, *d_codewordlens;
  unsigned int *d_cw32, *d_cw32len, *d_cw32idx, *d_cindex, *d_cindex2;
  s_compute = std::chrono::high_resolution_clock::now();
  CUDA_SAFE_CALL(hipMalloc((void **)&d_sourceData, mem_size));
  CUDA_SAFE_CALL(hipMalloc((void **)&d_destData, mem_size));
  CUDA_SAFE_CALL(hipMalloc((void **)&d_destDataPacked, mem_size));

  CUDA_SAFE_CALL(
      hipMalloc((void **)&d_codewords, NUM_SYMBOLS * symbol_type_size));
  CUDA_SAFE_CALL(
      hipMalloc((void **)&d_codewordlens, NUM_SYMBOLS * symbol_type_size));

  CUDA_SAFE_CALL(hipMalloc((void **)&d_cw32, mem_size));
  CUDA_SAFE_CALL(hipMalloc((void **)&d_cw32len, mem_size));
  CUDA_SAFE_CALL(hipMalloc((void **)&d_cw32idx, mem_size));

  CUDA_SAFE_CALL(
      hipMalloc((void **)&d_cindex, num_blocks * sizeof(unsigned int)));
  CUDA_SAFE_CALL(
      hipMalloc((void **)&d_cindex2, num_blocks * sizeof(unsigned int)));

  CUDA_SAFE_CALL(
      hipMemcpy(d_sourceData, sourceData, mem_size, hipMemcpyHostToDevice));
  CUDA_SAFE_CALL(hipMemcpy(d_codewords, codewords,
                            NUM_SYMBOLS * symbol_type_size,
                            hipMemcpyHostToDevice));
  CUDA_SAFE_CALL(hipMemcpy(d_codewordlens, codewordlens,
                            NUM_SYMBOLS * symbol_type_size,
                            hipMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      hipMemcpy(d_destData, destData, mem_size, hipMemcpyHostToDevice));

  dim3 grid_size(num_blocks, 1, 1);
  dim3 block_size(num_block_threads, 1, 1);
  unsigned int sm_size;

  unsigned int NT = 100000; // number of runs for each execution time

  unsigned int refbytesize;
  unsigned int num_ints = refbytesize / 4 + ((refbytesize % 4 == 0) ? 0 : 1);

  //////////////////* SM64HUFF KERNEL *///////////////////////////////////
  grid_size.x = num_blocks;
  block_size.x = num_block_threads;
  sm_size = block_size.x * sizeof(unsigned int);
#ifdef CACHECWLUT
  sm_size = 2 * NUM_SYMBOLS * sizeof(int) + block_size.x * sizeof(unsigned int);
#endif
  for (int i = 0; i < NT; i++) {
    vlc_encode_kernel_sm64huff<<<grid_size, block_size, sm_size>>>(
        d_sourceData, d_codewords, d_codewordlens, d_cw32, d_cw32len, d_cw32idx,
        d_destData, d_cindex);
  }
  hipDeviceSynchronize();

  CUT_CHECK_ERROR("Kernel execution failed\n");
  //////////////////* END KERNEL *///////////////////////////////////
  e_compute = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
#endif

#ifdef OUTPUT
  // Ensure the GPU data is copied back to host memory
  CUDA_SAFE_CALL(
      hipMemcpy(destData, d_destData, mem_size, hipMemcpyDeviceToHost));
  // Writing to a file
  FILE *fpo = fopen("result.txt", "w");
  if (fpo == NULL) {
    fprintf(stderr, "Failed to open file for writing.\n");
    exit(1); // Exit or handle the error appropriately
  }
  // Assuming destData is of type uint* and you want to print each element
  int i;
  for (i = 0; i < mem_size / sizeof(uint);
       i++) { // adjust the loop according to data type size
    fprintf(fpo, "%u\n",
            destData[i]); // Modify the format specifier based on your data type
  }
  fclose(fpo);

#endif

  free(sourceData);
  free(destData);
  free(codewords);
  free(codewordlens);
  free(cw32);
  free(cw32len);
  free(crefData);
  CUDA_SAFE_CALL(hipFree(d_sourceData));
  CUDA_SAFE_CALL(hipFree(d_destData));
  CUDA_SAFE_CALL(hipFree(d_destDataPacked));
  CUDA_SAFE_CALL(hipFree(d_codewords));
  CUDA_SAFE_CALL(hipFree(d_codewordlens));
  CUDA_SAFE_CALL(hipFree(d_cw32));
  CUDA_SAFE_CALL(hipFree(d_cw32len));
  CUDA_SAFE_CALL(hipFree(d_cw32idx));
  CUDA_SAFE_CALL(hipFree(d_cindex));
  CUDA_SAFE_CALL(hipFree(d_cindex2));
  free(cindex2);
}
