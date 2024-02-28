#define LIMIT -999
#include "needle.h"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes, kernels
#include "needle_kernel.cu"
#define WARMUP
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;

//#define BREAKDOWNS

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

// declaration, forward
void runTest(int argc, char **argv);

int maximum_cpu(int a, int b, int c) {
  int k;
  if (a <= b)
    k = b;
  else
    k = a;

  if (k <= c)
    return (c);
  else
    return (k);
}

int blosum62[24][24] = {{4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
                         -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4},
                        {-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
                         -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4},
                        {-2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
                         -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4},
                        {-2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
                         -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4},
                        {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
                         -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
                         0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4},
                        {-1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
                         -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
                         -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
                         -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
                         1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
                         2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},
                        {-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
                         -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4},
                        {-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
                         5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
                         0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
                         -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
                         -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4},
                        {0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
                         -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
                         -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
                         -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
                         1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4},
                        {-2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
                         -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4},
                        {-1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
                         -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  runTest(argc, argv);

  return EXIT_SUCCESS;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
  exit(1);
}

void runTest(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  int max_rows, max_cols, penalty;
  int *input_itemsets, *output_itemsets, *referrence;
  int *matrix_cuda, *referrence_cuda;
  int size;

  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  if (argc == 3) {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
  } else {
    usage(argc, argv);
  }

  if (atoi(argv[1]) % 16 != 0) {
    fprintf(stderr, "The dimension values must be a multiple of 16\n");
    exit(1);
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
  input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
  output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

  if (!input_itemsets)
    fprintf(stderr, "error: can not allocate memory");
#ifdef RAND
  srand(7);
#endif
  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  for (int i = 1; i < max_rows; i++) { // please define your own sequence.
    input_itemsets[i * max_cols] = i % 10 + 1;
#ifdef RAND
    input_itemsets[i * max_cols] = rand() % 10 + 1;
#endif
  }
  for (int j = 1; j < max_cols; j++) { // please define your own sequence.
    input_itemsets[j] = j % 10 + 1;
#ifdef RAND
    input_itemsets[j] = rand() % 10 + 1;
#endif
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      referrence[i * max_cols + j] =
          blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (int i = 1; i < max_rows; i++)
    input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; j++)
    input_itemsets[j] = -j * penalty;

  size = max_cols * max_rows;
  auto end_0 = std::chrono::high_resolution_clock::now();
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  char *warm;
  cudaMalloc((void **)&warm, sizeof(char));
  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  s_compute = std::chrono::high_resolution_clock::now();
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  cudaMalloc((void **)&referrence_cuda, sizeof(int) * size);
  cudaMalloc((void **)&matrix_cuda, sizeof(int) * size);

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size,
             cudaMemcpyHostToDevice);
#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  dim3 dimGrid;
  dim3 dimBlock(BLOCK_SIZE, 1);
  int block_width = (max_cols - 1) / BLOCK_SIZE;

  // process top-left matrix
  for (int i = 1; i <= block_width; i++) {
    dimGrid.x = i;
    dimGrid.y = 1;
    needle_cuda_shared_1<<<dimGrid, dimBlock>>>(
        referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
#ifdef DEBUG
    int sz = max_cols * max_cols;
    int *referrence_h = (int *)malloc(sz * sizeof(int));
    int *matrix_h = (int *)malloc(sz * sizeof(int));

    cudaMemcpy(referrence_h, referrence_cuda, sz * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(matrix_h, matrix_cuda, sz * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 2; i++) {
      std::cerr << "i: " << i << " m = " << matrix_h[i]
                << ", ref = " << referrence_h[i] << std::endl;
    }
#endif
  }

  // process bottom-right matrix
  for (int i = block_width - 1; i >= 1; i--) {
    dimGrid.x = i;
    dimGrid.y = 1;
    needle_cuda_shared_2<<<dimGrid, dimBlock>>>(
        referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
#ifdef DEBUG
    int sz = max_cols * max_cols;
    int *referrence_h = (int *)malloc(sz * sizeof(int));
    int *matrix_h = (int *)malloc(sz * sizeof(int));

    cudaMemcpy(referrence_h, referrence_cuda, sz * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(matrix_h, matrix_cuda, sz * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 2048; i < 2052; i++) {
      std::cerr << "i: " << i << " m = " << matrix_h[i]
                << ", ref = " << referrence_h[i] << std::endl;
    }
#endif
  }
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size,
             cudaMemcpyDeviceToHost);
  /*for (int i = 2048; i < 2052; i++) {
    std::cerr << "i: " << i << " m = " << output_itemsets[i] << std::endl;
  }
  std::cerr << "size" << size << std::endl;
*/
#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif

//#define TRACEBACK
#ifdef OUTPUT
  FILE *fpo = fopen("nat_result.txt", "w");
  fprintf(fpo, "print traceback value GPU:\n");

  for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;) {
    int nw, n, w, traceback;
    if (i == max_rows - 2 && j == max_rows - 2)
      fprintf(fpo, "%d ",
              output_itemsets[i * max_cols + j]); // print the first element
    if (i == 0 && j == 0)
      break;
    if (i > 0 && j > 0) {
      nw = output_itemsets[(i - 1) * max_cols + j - 1];
      w = output_itemsets[i * max_cols + j - 1];
      n = output_itemsets[(i - 1) * max_cols + j];
    } else if (i == 0) {
      nw = n = LIMIT;
      w = output_itemsets[i * max_cols + j - 1];
    } else if (j == 0) {
      nw = w = LIMIT;
      n = output_itemsets[(i - 1) * max_cols + j];
    } else {
    }

    // traceback = maximum_cpu(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + referrence[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum_cpu(new_nw, new_w, new_n);
    if (traceback == new_nw)
      traceback = nw;
    if (traceback == new_w)
      traceback = w;
    if (traceback == new_n)
      traceback = n;

    fprintf(fpo, "%d ", traceback);

    if (traceback == nw) {
      i--;
      j--;
      continue;
    }

    else if (traceback == w) {
      j--;
      continue;
    }

    else if (traceback == n) {
      i--;
      continue;
    }

    else
      ;
  }

  fclose(fpo);

#endif

  cudaFree(referrence_cuda);
  cudaFree(matrix_cuda);
  e_compute = std::chrono::high_resolution_clock::now();

  free(referrence);
  free(input_itemsets);
  free(output_itemsets);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli_0 = end_0 - start_0;
  std::cerr << "Init time: " << elapsed_milli_0.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

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

  std::chrono::duration<double, std::milli> elapsed_milli = end - start;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;

#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
  // free warmup
  cudaFree(warm);
#endif
}
