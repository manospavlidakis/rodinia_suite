#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include "bucketsort.cuh"
#include "helper_cuda.h"
#include "helper_timer.h"
#include "mergesort.cuh"
#include <chrono>
#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
#define WARMUP
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;
////////////////////////////////////////////////////////////////////////////////
// Size of the testset (Bitwise shift of 1 over 22 places)
////////////////////////////////////////////////////////////////////////////////
#define SIZE (1 << 22)
////////////////////////////////////////////////////////////////////////////////
// Number of tests to average over
////////////////////////////////////////////////////////////////////////////////
#define TEST 1000

////////////////////////////////////////////////////////////////////////////////
// Compare method for CPU sort
////////////////////////////////////////////////////////////////////////////////
inline int compare(const void *a, const void *b) {
  if (*((float *)a) < *((float *)b))
    return -1;
  else if (*((float *)a) > *((float *)b))
    return 1;
  else
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
// Forward declaration
////////////////////////////////////////////////////////////////////////////////
void cudaSort(float *origList, float minimum, float maximum, float *resultList,
              int numElements);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  auto start_all = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  int numElements = 0;
  // Number of elements in the test bed
  if (strcmp(argv[1], "r") == 0) {
    numElements = SIZE;
  } else {
    FILE *fp;
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
      cout << "Error reading file" << endl;
      exit(EXIT_FAILURE);
    }
    int count = 0;
    float c;

    while (fscanf(fp, "%f", &c) != EOF) {
      count++;
    }
    fclose(fp);

    numElements = count;
  }
  // cout << "Sorting list of " << numElements << " floats\n";
  //  Generate random data
  //  Memory space the list of random floats will take up
  int mem_size = numElements * sizeof(float);
  // Allocate enough for the input list
  float *cpu_idata = (float *)malloc(mem_size);
  // Allocate enough for the output list on the cpu side
  float *cpu_odata = (float *)malloc(mem_size);
  // Allocate enough memory for the output list on the gpu side
  float *gpu_odata = (float *)malloc(mem_size);

  float datamin = FLT_MAX;
  float datamax = -FLT_MAX;
  if (strcmp(argv[1], "r") == 0) {
    for (int i = 0; i < numElements; i++) {
      // Generate random floats between 0 and 1 for the input data
      cpu_idata[i] = ((float)rand() / RAND_MAX);
      // Compare data at index to data minimum, if less than current minimum,
      // set that element as new minimum
      datamin = min(cpu_idata[i], datamin);
      // Same as above but for maximum
      datamax = max(cpu_idata[i], datamax);
    }

  } else {
    FILE *fp;
    fp = fopen(argv[1], "r");
    for (int i = 0; i < numElements; i++) {
      fscanf(fp, "%f", &cpu_idata[i]);
      datamin = min(cpu_idata[i], datamin);
      datamax = max(cpu_idata[i], datamax);
    }
  }
  auto end_0 = std::chrono::high_resolution_clock::now();
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  double *warm;
  cudaMalloc((void **)&warm, sizeof(double) * 100000);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaFree(warm);
  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  s_compute = std::chrono::high_resolution_clock::now();
  // cout << "Sorting on GPU..." << flush;
  // GPU Sort
  for (int i = 0; i < TEST; i++)
    cudaSort(cpu_idata, datamin, datamax, gpu_odata, numElements);
  // cout << "done.\n";
  e_compute = std::chrono::high_resolution_clock::now();
  // Timer report
  // printf("GPU iterations: %d\n", TEST);

#ifdef OUTPUT
  FILE *tp;
  const char filename2[] = "./result.txt";
  tp = fopen(filename2, "w");
  for (int i = 0; i < numElements; i++) {
    fprintf(tp, "%f ", cpu_idata[i]);
  }

  fclose(tp);
#endif

  free(cpu_idata);
  free(cpu_odata);
  free(gpu_odata);
  auto end_all = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli_0 = end_0 - start_0;
  std::cerr << "Init time: " << elapsed_milli_0.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> elapsed_milli = end_all - start_all;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
#endif
}

void cudaSort(float *origList, float minimum, float maximum, float *resultList,
              int numElements) {
  // Initialization and upload data
  float *d_input = NULL;
  float *d_output = NULL;
  int mem_size = (numElements + DIVISIONS * 4) * sizeof(float);

  cudaMalloc((void **)&d_input, mem_size);
  cudaMalloc((void **)&d_output, mem_size);
  cudaMemcpy((void *)d_input, (void *)origList, numElements * sizeof(float),
             cudaMemcpyHostToDevice);
  init_bucketsort(numElements);

  int *sizes = (int *)malloc(DIVISIONS * sizeof(int));
  int *nullElements = (int *)malloc(DIVISIONS * sizeof(int));
  unsigned int *origOffsets =
      (unsigned int *)malloc((DIVISIONS + 1) * sizeof(int));
  bucketSort(d_input, d_output, numElements, sizes, nullElements, minimum,
             maximum, origOffsets);

  float4 *d_origList = (float4 *)d_output, *d_resultList = (float4 *)d_input;
  int newlistsize = 0;

  for (int i = 0; i < DIVISIONS; i++)
    newlistsize += sizes[i] * 4;

  float4 *mergeresult =
      runMergeSort(newlistsize, DIVISIONS, d_origList, d_resultList, sizes,
                   nullElements, origOffsets); // d_origList;
  cudaThreadSynchronize();

  checkCudaErrors(cudaMemcpy((void *)resultList, (void *)mergeresult,
                             numElements * sizeof(float),
                             cudaMemcpyDeviceToHost));

  // Clean up
  finish_bucketsort();
  cudaFree(d_input);
  cudaFree(d_output);
  free(nullElements);
  free(sizes);
}
