/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include "OpenCL.h"
#include "pathfinder_common.h"
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#define DEBUG
#include "../common/opencl_util.h"
std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
using namespace std;
#define HALO                                                                   \
  1 // halo width along one direction when advancing to the next iteration
#define DEVICE 0
#define M_SEED 9
//#define BENCH_PRINT
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

// Program variables.
int rows, cols;
int Ne = rows * cols;
int *data;
int **wall;
int *result;
int pyramid_height;
FILE *fpo;
char *ofile = NULL;

void init(int argc, char **argv) {
  if (argc == 4 || argc == 5) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
    if (argc == 5) {
      ofile = argv[4];
    }
  } else {
    printf("Usage: %s row_len col_len pyramid_height output_file\n", argv[0]);
    exit(0);
  }
  data = (int *)alignedMalloc(rows * cols * sizeof(int));
  wall = (int **)alignedMalloc(rows * sizeof(int *));
  for (int n = 0; n < rows; n++) {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }
  result = (int *)alignedMalloc(cols * sizeof(int));

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }

#ifdef DEBUG
  fpo = fopen("result.txt", "w");
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      fprintf(fpo, "%d ", wall[i][j]);
    }
    fprintf(fpo, "\n");
  }
  fprintf(fpo, "--------------------------------------------------\n");
#endif
}

void fatal(char *s) { fprintf(stderr, "error: %s\n", s); }

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  int version = 0;

  init_fpga(&argc, &argv, &version);
  version = 0;
  init(argc, argv);
  // Pyramid parameters.
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BSIZE - (pyramid_height)*HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  int size = rows * cols;
  // Create and initialize the OpenCL object.
  s_init_fpga_timer = std::chrono::high_resolution_clock::now();
  OpenCL cl(0); // 1 means to display output (debugging mode).
  cl.init(version);
  cl.gwSize(BSIZE * blockCols);
  // cl.lwSize(BSIZE);
  cl.lwSize(128);

  // Create and build the kernel.
  string kn = "dynproc_kernel"; // the kernel name, for future use.
  cl.createKernel(kn);
  e_init_fpga_timer = std::chrono::high_resolution_clock::now();

  auto end_0 = std::chrono::high_resolution_clock::now();

  s_compute = std::chrono::high_resolution_clock::now();
  // Allocate device memory.
  cl_mem d_gpuWall;
  cl_mem d_gpuResult[2];
  cl_int error;
  d_gpuWall = clCreateBuffer(cl.ctxt(), CL_MEM_READ_ONLY,
                             sizeof(cl_int) * (size - cols), NULL, &error);
  if (error != CL_SUCCESS) {
    printf("Failed to allocate device buffer!\n");
    exit(-1);
  }
  d_gpuResult[0] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE,
                                  sizeof(cl_int) * cols, NULL, &error);
  if (error != CL_SUCCESS) {
    printf("Failed to allocate device buffer!\n");
    exit(-1);
  }

  CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuWall, CL_TRUE, 0,
                                    sizeof(cl_int) * (size - cols), data + cols,
                                    0, NULL, NULL));
  CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuResult[0], CL_TRUE,
                                    0, sizeof(cl_int) * cols, data, 0, NULL,
                                    NULL));

  d_gpuResult[1] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE,
                                  sizeof(cl_int) * cols, NULL, &error);
  if (error != CL_SUCCESS) {
    printf("Failed to allocate device buffer!\n");
    exit(-1);
  }

  int src, dst;
  src = 1;
  dst = 0;

  // Set fixed kernel arguments
  CL_SAFE_CALL(
      clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void *)&d_gpuWall));
  CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 4, sizeof(cl_int), (void *)&cols));
  CL_SAFE_CALL(
      clSetKernelArg(cl.kernel(kn), 6, sizeof(cl_int), (void *)&borderCols));

  for (int startStep = 0; startStep < rows - 1; startStep += pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;

    // Calculate changed kernel arguments...
    int iteration = MIN(pyramid_height, rows - startStep - 1);
    CL_SAFE_CALL(
        clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_int), (void *)&iteration));
    CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_mem),
                                (void *)&d_gpuResult[src]));
    CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_mem),
                                (void *)&d_gpuResult[dst]));
    CL_SAFE_CALL(
        clSetKernelArg(cl.kernel(kn), 5, sizeof(cl_int), (void *)&startStep));

    // Launch kernel
    cl.launch(kn, version);
  }

  // Copy results back to host.
  clEnqueueReadBuffer(cl.q(), d_gpuResult[dst], CL_TRUE, 0,
                      sizeof(cl_int) * cols, result, 0, NULL, NULL);
  e_compute = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
  /*  for (int i = 0; i < cols; i++)
      fprintf(fpo, "%d ", data[i]);
    fprintf(fpo, "\n");
  */
  for (int i = 0; i < cols; i++)
    fprintf(fpo, "%d", result[i]);
  fprintf(fpo, "\n");
  fclose(fpo);
#endif
  // Memory cleanup here.
  free(data);
  free(wall);
  free(result);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli = end - start;
  std::chrono::duration<double, std::milli> prep_milli =
      e_init_fpga_timer - s_init_fpga_timer;
  std::chrono::duration<double, std::milli> elapsed_milli_0 = end_0 - start_0;
  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "FPGA time: " << prep_milli.count() << " ms" << std::endl;
  std::cerr << "Init time (withoutFPGA): "
            << elapsed_milli_0.count() - prep_milli.count() << " ms"
            << std::endl;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;
  std::cerr << "Elapsed time (withoutFPGA): "
            << elapsed_milli.count() - prep_milli.count() << " ms" << std::endl;

  return EXIT_SUCCESS;
}
