/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.
 **   The sequential version is gaussian.c.  This parallel
 **   implementation converts three independent for() loops
 **   into three Fans.  Use the data file ge_3.dat to verify
 **   the correction of the output.
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include "cuda.h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BREAKDOWNS
#define RED "\033[1;31m"
#define RESET "\033[0m"
#define CUDA_ERROR_FATAL_RUNTIME(err)                                          \
  cudaErrorCheckFatal_Runtime(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
cudaErrorCheckFatal_Runtime(cudaError_t err, const char *func, const char *file,
                            size_t line) {
  if (err != cudaSuccess) {
    std::cerr << RED << func << " error : " << RESET << cudaGetErrorString(err)
              << std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
}


std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;
#define DEBUG
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

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else
#define MAXBLOCKSIZE 512
#endif

// 2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif
#define WARMUP
int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
__global__ void Fan1(float *m, float *a, int Size, int t);
__global__ void Fan2(float *m, float *a, float *b, int Size, int j1, int t);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
  int i, j;
  float lamda = -0.01;
  float coe[2 * size - 1];
  float coe_i = 0.0;

  for (i = 0; i < size; i++) {
    coe_i = 10 * exp(lamda * i);
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      m[i * size + j] = coe[size - 1 - i + j];
    }
  }
}

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();

  int i, j;
  char flag;
  if (argc < 2) {
    printf("Usage: gaussian -f filename / -s size [-q]\n\n");
    printf("-q (quiet) suppresses printing the matrix and result values.\n");
    printf("-f (filename) path of input file\n");
    printf(
        "-s (size) size of matrix. Create matrix and rhs in this program \n");
    printf(
        "The first line of the file contains the dimension of the matrix, n.");
    printf("The second line of the file is a newline.\n");
    printf("The next n lines contain n tab separated values for the matrix.");
    printf("The next line of the file is a newline.\n");
    printf("The next line of the file is a 1xn vector with tab separated "
           "values.\n");
    printf("The next line of the file is a newline. (optional)\n");
    printf("The final line of the file is the pre-computed solution. "
           "(optional)\n");
    printf("Example: matrix4.txt:\n");
    printf("4\n");
    printf("\n");
    printf("-0.6	-0.5	0.7	0.3\n");
    printf("-0.3	-0.9	0.3	0.7\n");
    printf("-0.4	-0.5	-0.3	-0.8\n");
    printf("0.0	-0.1	0.2	0.9\n");
    printf("\n");
    printf("-0.85	-0.68	0.24	-0.53\n");
    printf("\n");
    printf("0.7	0.0	-0.4	-0.5\n");
    exit(0);
  }
  // PrintDeviceProperties();
  // char filename[100];
  // sprintf(filename,"matrices/matrix%d.txt",size);

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 's': // platform
        i++;
        Size = atoi(argv[i]);
        printf("Create matrix internally in parse, size = %d \n", Size);

        a = (float *)malloc(Size * Size * sizeof(float));
        create_matrix(a, Size);

        b = (float *)malloc(Size * sizeof(float));
        for (j = 0; j < Size; j++)
          b[j] = 1.0;

        m = (float *)malloc(Size * Size * sizeof(float));
        break;
      case 'f': // platform
        i++;
        InitProblemOnce(argv[i]);
        break;
      }
    }
  }
  // create a new vector to hold the final answer
  finalVec = (float *)malloc(Size * sizeof(float));
  // InitProblemOnce(filename);
  InitPerRun();
  auto end_0 = std::chrono::high_resolution_clock::now();

#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  char *warm;
  cudaMalloc((void **)&warm, sizeof(char));
  end_warmup = std::chrono::high_resolution_clock::now();
#endif

  s_compute = std::chrono::high_resolution_clock::now();
  // run kernels
  ForwardSub();

#ifdef INPUT
  printf("Matrix m is: \n");
  PrintMat(m, Size, Size);

  printf("Matrix a is: \n");
  PrintMat(a, Size, Size);

  printf("Array b is: \n");
  PrintAry(b, Size);
#endif

  BackSub();

#ifdef OUTPUT
  printf("The final solution is: \n");
  PrintAry(finalVec, Size);
#endif
  e_compute = std::chrono::high_resolution_clock::now();

  free(m);
  free(a);
  free(b);
  free(finalVec);
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
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties() {
  cudaDeviceProp deviceProp;
  int nDevCount = 0;

  cudaGetDeviceCount(&nDevCount);
  printf("Total Device found: %d", nDevCount);
  for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx) {
    memset(&deviceProp, 0, sizeof(deviceProp));
    if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx)) {
      printf("\nDevice Name \t\t - %s ", deviceProp.name);
      printf("\n**************************************");
      printf("\nTotal Global Memory\t\t\t - %lu KB",
             deviceProp.totalGlobalMem / 1024);
      printf("\nShared memory available per block \t - %lu KB",
             deviceProp.sharedMemPerBlock / 1024);
      printf("\nNumber of registers per thread block \t - %d",
             deviceProp.regsPerBlock);
      printf("\nWarp size in threads \t\t\t - %d", deviceProp.warpSize);
      printf("\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch);
      printf("\nMaximum threads per block \t\t - %d",
             deviceProp.maxThreadsPerBlock);
      printf("\nMaximum Thread Dimension (block) \t - %d %d %d",
             deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
             deviceProp.maxThreadsDim[2]);
      printf("\nMaximum Thread Dimension (grid) \t - %d %d %d",
             deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
             deviceProp.maxGridSize[2]);
      printf("\nTotal constant memory \t\t\t - %zu bytes",
             deviceProp.totalConstMem);
      printf("\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major, deviceProp.minor);
      printf("\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate);
      printf("\nTexture Alignment \t\t\t - %zu bytes",
             deviceProp.textureAlignment);
      printf("\nDevice Overlap \t\t\t\t - %s",
             deviceProp.deviceOverlap ? "Allowed" : "Not Allowed");
      printf("\nNumber of Multi processors \t\t - %d\n\n",
             deviceProp.multiProcessorCount);
    } else
      printf("\n%s", cudaGetErrorString(cudaGetLastError()));
  }
}

/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename) {
  // char *filename = argv[1];

  // printf("Enter the data file name: ");
  // scanf("%s", filename);
  // printf("The file name is: %s\n", filename);

  fp = fopen(filename, "r");

  fscanf(fp, "%d", &Size);

  a = (float *)malloc(Size * Size * sizeof(float));

  InitMat(a, Size, Size);
  // printf("The input matrix a is:\n");
  // PrintMat(a, Size, Size);
  b = (float *)malloc(Size * sizeof(float));

  InitAry(b, Size);
  // printf("The input array b is:\n");
  // PrintAry(b, Size);

  m = (float *)malloc(Size * Size * sizeof(float));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() {
  int i;
  for (i = 0; i < Size * Size; i++)
    *(m + i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t) {
  // if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
  // printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

  if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
    return;
  *(m_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) =
      *(a_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) /
      *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size,
                     int j1, int t) {
  if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
    return;
  if (threadIdx.y + blockIdx.y * blockDim.y >= Size - t)
    return;

  int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  int yidx = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

  a_cuda[Size * (xidx + 1 + t) + (yidx + t)] -=
      m_cuda[Size * (xidx + 1 + t) + t] * a_cuda[Size * t + (yidx + t)];
  // a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
  if (yidx == 0) {
    // printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
    // printf("xidx:%d,yidx:%d\n",xidx,yidx);
    b_cuda[xidx + 1 + t] -=
        m_cuda[Size * (xidx + 1 + t) + (yidx + t)] * b_cuda[t];
  }
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub() {
  int t;
  float *m_cuda, *a_cuda, *b_cuda;
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  // allocate memory on GPU
  cudaMalloc((void **)&m_cuda, Size * Size * sizeof(float));
  cudaMalloc((void **)&a_cuda, Size * Size * sizeof(float));
  cudaMalloc((void **)&b_cuda, Size * sizeof(float));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  // copy memory to GPU
  CUDA_ERROR_FATAL_RUNTIME(cudaMemcpy(m_cuda, m, Size * Size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_ERROR_FATAL_RUNTIME(cudaMemcpy(a_cuda, a, Size * Size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_ERROR_FATAL_RUNTIME(cudaMemcpy(b_cuda, b, Size * sizeof(float), cudaMemcpyHostToDevice));

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  double data_size = (2 * (Size * Size * sizeof(float))) + Size * sizeof(float);
  std::cerr << std::fixed << " Data size: " << data_size / 1024 / 1024 << " MB"
            << std::endl;

  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  int block_size, grid_size;

  block_size = MAXBLOCKSIZE;
  grid_size = (Size / block_size) + (!(Size % block_size) ? 0 : 1);
  // printf("1d grid size: %d\n",grid_size);

  dim3 dimBlock(block_size);
  dim3 dimGrid(grid_size);
  // dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

  int blockSize2d, gridSize2d;
  blockSize2d = BLOCK_SIZE_XY;
  gridSize2d = (Size / blockSize2d) + (!(Size % blockSize2d ? 0 : 1));

  dim3 dimBlockXY(blockSize2d, blockSize2d);
  dim3 dimGridXY(gridSize2d, gridSize2d);

  for (t = 0; t < (Size - 1); t++) {
    Fan1<<<dimGrid, dimBlock>>>(m_cuda, a_cuda, Size, t);
    Fan2<<<dimGridXY, dimBlockXY>>>(m_cuda, a_cuda, b_cuda, Size, Size - t, t);
    // cudaThreadSynchronize();
    checkCUDAError("Fan2");
  }
  CUDA_ERROR_FATAL_RUNTIME(cudaDeviceSynchronize());
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  // copy memory back to CPU
  CUDA_ERROR_FATAL_RUNTIME(cudaMemcpy(m, m_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_ERROR_FATAL_RUNTIME(cudaMemcpy(a, a_cuda, Size * Size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_ERROR_FATAL_RUNTIME(cudaMemcpy(b, b_cuda, Size * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif

  cudaFree(m_cuda);
  cudaFree(a_cuda);
  cudaFree(b_cuda);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub() {
  // solve "bottom up"
  int i, j;
  for (i = 0; i < Size; i++) {
    finalVec[Size - i - 1] = b[Size - i - 1];
    for (j = 0; j < i; j++) {
      finalVec[Size - i - 1] -= *(a + Size * (Size - i - 1) + (Size - j - 1)) *
                                finalVec[Size - j - 1];
    }
    finalVec[Size - i - 1] =
        finalVec[Size - i - 1] / *(a + Size * (Size - i - 1) + (Size - i - 1));
  }
}

void InitMat(float *ary, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      fscanf(fp, "%f", ary + Size * i + j);
    }
  }
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      printf("%8.2f ", *(ary + Size * i + j));
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size) {
  int i;

  for (i = 0; i < ary_size; i++) {
    fscanf(fp, "%f", &ary[i]);
  }
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size) {
  std::cerr << " Store results to output!!" << std::endl;
  // Store the result into a file.
  FILE *fpo = fopen("nat_result.txt", "w");

  int i;
  for (i = 0; i < ary_size; i++) {
    fprintf(fpo, "%.2f", ary[i]);
  }
  fclose(fpo);
}
/*void PrintAry(float *ary, int ary_size) {
  int i;
  for (i = 0; i < ary_size; i++) {
    printf("%.2f ", ary[i]);
  }
  printf("\n\n");
}*/
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
