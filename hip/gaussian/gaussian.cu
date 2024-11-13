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
#include "hip/hip_runtime.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;
#define WARMUP
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
void VerifyResult();

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

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 's': // platform
        i++;
        Size = atoi(argv[i]);

        a = (float *)malloc(Size * Size * sizeof(float));
        create_matrix(a, Size);

        b = (float *)malloc(Size * sizeof(float));
        for (j = 0; j < Size; j++)
          b[j] = 1.0;

        m = (float *)malloc(Size * Size * sizeof(float));
        break;
      case 'f': // platform
        i++;
        printf("Read file from %s \n", argv[i]);
        InitProblemOnce(argv[i]);
        break;
      }
    }
  }

  finalVec = (float *)malloc(Size * sizeof(float));
  // InitProblemOnce(filename);
  InitPerRun();
  auto end_0 = std::chrono::high_resolution_clock::now();

#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  char *warm;
  hipMalloc((void **)&warm, sizeof(char));
  hipStream_t stream;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

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
//  VerifyResult();
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

  std::chrono::duration<double, std::milli> elapsed_milli = end - start;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
  // free warmup
  hipFree(warm);
#endif
}
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties() {
  hipDeviceProp_t deviceProp;
  int nDevCount = 0;

  hipGetDeviceCount(&nDevCount);
  printf("Total Device found: %d", nDevCount);
  for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx) {
    memset(&deviceProp, 0, sizeof(deviceProp));
    if (hipSuccess == hipGetDeviceProperties(&deviceProp, nDeviceIdx)) {
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
      printf("\nNumber of Multi processors \t\t - %d\n\n",
             deviceProp.multiProcessorCount);
    } else
      printf("\n%s", hipGetErrorString(hipGetLastError()));
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
  // if(hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x >= Size-1-t) printf(".");
  // printf("blockIDx.x:%d,hipThreadIdx_x:%d,Size:%d,t:%d,Size-1-t:%d\n",hipBlockIdx_x,hipThreadIdx_x,Size,t,Size-1-t);

  if (hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x >= Size - 1 - t)
    return;
  *(m_cuda + Size * (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + t + 1) +
    t) =
      *(a_cuda +
        Size * (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + t + 1) + t) /
      *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size,
                     int j1, int t) {
  if (hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x >= Size - 1 - t)
    return;
  if (hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y >= Size - t)
    return;

  int xidx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int yidx = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  // printf("hipBlockIdx_x:%d,hipThreadIdx_x:%d,hipBlockIdx_y:%d,hipThreadIdx_y:%d,hipBlockDim_x:%d,hipBlockDim_y:%d\n",hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y,hipThreadIdx_y,hipBlockDim_x,hipBlockDim_y);

  a_cuda[Size * (xidx + 1 + t) + (yidx + t)] -=
      m_cuda[Size * (xidx + 1 + t) + t] * a_cuda[Size * t + (yidx + t)];
  // a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
  if (yidx == 0) {
    // printf("hipBlockIdx_x:%d,hipThreadIdx_x:%d,hipBlockIdx_y:%d,hipThreadIdx_y:%d,hipBlockDim_x:%d,hipBlockDim_y:%d\n",hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y,hipThreadIdx_y,hipBlockDim_x,hipBlockDim_y);
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

  // allocate memory on GPU
  hipMalloc((void **)&m_cuda, Size * Size * sizeof(float));

  hipMalloc((void **)&a_cuda, Size * Size * sizeof(float));

  hipMalloc((void **)&b_cuda, Size * sizeof(float));

  // copy memory to GPU
  hipMemcpy(m_cuda, m, Size * Size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(a_cuda, a, Size * Size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(b_cuda, b, Size * sizeof(float), hipMemcpyHostToDevice);

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
    hipLaunchKernelGGL(Fan1, dim3(dimGrid), dim3(dimBlock), 0, 0, m_cuda,
                       a_cuda, Size, t);
    hipLaunchKernelGGL(Fan2, dim3(dimGridXY), dim3(dimBlockXY), 0, 0, m_cuda,
                       a_cuda, b_cuda, Size, Size - t, t);
    checkCUDAError("Fan2");
  }
  // copy memory back to CPU
  hipMemcpy(m, m_cuda, Size * Size * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(a, a_cuda, Size * Size * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(b, b_cuda, Size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(m_cuda);
  hipFree(a_cuda);
  hipFree(b_cuda);
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
void checkCUDAError(const char *msg) {
  hipError_t err = hipGetLastError();
  if (hipSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void VerifyResult() {
  int i, j;
  float tmp_out = 0;
  for (i = 0; i < Size; i++) {
    for (j = 0, tmp_out = 0; j < Size; j++)
      tmp_out += (*(a + Size * i + j) * finalVec[j]);
    if (abs(tmp_out - b[i]) > 0.01) {
      printf("Test Failed\n");
      printf("out[%d]: %f; b[%d]:%f; diff:%f\n", i, tmp_out, i, b[i],
             b[i] - tmp_out);
      return;
    }
  }

  printf("Test Pass\n");
  return;
}
