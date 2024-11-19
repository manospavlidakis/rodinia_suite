#include <assert.h>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1

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
#endif

void run(int argc, char **argv);

int rows, cols;
int *data;
int **wall;
int *result;
#define M_SEED 9
int pyramid_height;
FILE *fpo;

void init(int argc, char **argv) {
  if (argc == 4) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  } else {
    printf("Usage: dynproc row_len col_len pyramid_height\n");
    exit(0);
  }
  data = new int[rows * cols];
  wall = new int *[rows];
  for (int n = 0; n < rows; n++)
    wall[n] = data + cols * n;
  result = new int[cols];
#ifdef RAND
  int seed = M_SEED;
  srand(seed);
#endif
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = i * j % 10;
#ifdef RAND
      wall[i][j] = rand() % 10;
#endif
    }
  }
#ifdef OUTPUT
  std::cerr << "append to file!!!" << std::endl;
  fpo = fopen("nat_result.txt", "w");
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      fprintf(fpo, "%d ", wall[i][j]);
    }
    fprintf(fpo, "\n");
  }
#endif
}
void fatal(char *s) { fprintf(stderr, "error: %s\n", s); }

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc,
                               int *gpuResults, int cols, int rows,
                               int startStep, int border) {

  __shared__ int prev[BLOCK_SIZE];
  __shared__ int result[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX = small_block_cols * bx - border;
  int blkXmax = blkX + BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                       : BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, cols - 1)) {
    prev[tx] = gpuSrc[xidx];
  }
  __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    __syncthreads();
    if (i == iteration - 1)
      break;
    if (computed) // Assign the computation range
      prev[tx] = result[tx];
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols) {

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(blockCols);

  int src = 1, dst = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;
    dynproc_kernel<<<dimGrid, dimBlock>>>(
        MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
        gpuResult[dst], cols, rows, t, borderCols);
  }
  return dst;
}
int main(int argc, char **argv) {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if (num_devices > 1)
    cudaSetDevice(DEVICE);

  run(argc, argv);

  return EXIT_SUCCESS;
}

void run(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  init(argc, argv);

  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);
  int size = rows * cols;
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

  int *gpuWall, *gpuResult[2];

  cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols);
  cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols);
  cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols));
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
             cudaMemcpyHostToDevice);
#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  /* double input_size = 2 * sizeof(int) * cols;
   double output_size = sizeof(int) * (size - cols);
   std::cerr << "Input size: " << input_size << std::endl;
   std::cerr << "Output size: " << output_size << std::endl;*/
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height,
                            blockCols, borderCols);

  cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
             cudaMemcpyDeviceToHost);
#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
#endif

#ifdef OUTPUT
  for (int i = 0; i < cols; i++)
    fprintf(fpo, "%d ", data[i]);
  fprintf(fpo, "\n");

  for (int i = 0; i < cols; i++)
    fprintf(fpo, "%d ", result[i]);
  fprintf(fpo, "\n");
  fclose(fpo);
#endif

  cudaFree(gpuWall);
  cudaFree(gpuResult[0]);
  cudaFree(gpuResult[1]);
  e_compute = std::chrono::high_resolution_clock::now();

  delete[] data;
  delete[] wall;
  delete[] result;
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
