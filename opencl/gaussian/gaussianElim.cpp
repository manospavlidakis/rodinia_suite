#ifndef __GAUSSIAN_ELIMINATION__
#define __GAUSSIAN_ELIMINATION__
#include "gaussianElim.h"
#include <chrono>
#include <math.h>
//#define DEBUG
#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_0 RD_WG_SIZE
#else
#define BLOCK_SIZE_0 0
#endif

// 2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_1_X RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_1_X RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_1_X RD_WG_SIZE
#else
#define BLOCK_SIZE_1_X 0
#endif

#ifdef RD_WG_SIZE_1_1
#define BLOCK_SIZE_1_Y RD_WG_SIZE_1_1
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_1_Y RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_1_Y RD_WG_SIZE
#else
#define BLOCK_SIZE_1_Y 0
#endif
cl_context context = NULL;
cl_kernel fan1_kernel, fan2_kernel;
cl_program gaussianElim_program;

std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;

float *a, *b, *finalVec;
float *m;
int size;
FILE *fp;

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

  // args
  char filename[200];
  int quiet = 1, timing = 0, platform = 1, device = 0;

  // parse command line
  if (parseCommandline(argc, argv, filename, &quiet, &timing, &platform,
                       &device, &size)) {
    printUsage();
    return 0;
  }
  s_init_fpga_timer = std::chrono::high_resolution_clock::now();

  context = cl_init_context(platform, device, quiet);
  // 1. set up kernels
  cl_int status = 0;

  gaussianElim_program =
      cl_compileProgram((char *)"gaussianElim_kernels.aocx", NULL);

  fan1_kernel = clCreateKernel(gaussianElim_program, "Fan1", &status);
  status = cl_errChk(status, (char *)"Error Creating Fan1 kernel", true);
  if (status)
    exit(1);

  fan2_kernel = clCreateKernel(gaussianElim_program, "Fan2", &status);
  status = cl_errChk(status, (char *)"Error Creating Fan2 kernel", true);
  if (status)
    exit(1);

  e_init_fpga_timer = std::chrono::high_resolution_clock::now();

  fp = fopen(filename, "r");
  fscanf(fp, "%d", &size);

  a = (float *)malloc(size * size * sizeof(float));
  InitMat(fp, size, a, size, size);

  b = (float *)malloc(size * sizeof(float));
  InitAry(fp, b, size);

  m = (float *)malloc(size * size * sizeof(float));

  fclose(fp);

#ifdef DEBUG
  printf("The input matrix a is:\n");
  PrintMat(a, size, size, size);

  printf("The input array b is:\n");
  PrintAry(b, size);
#endif
  // create the solution matrix

  finalVec = (float *)malloc(size * sizeof(float));

  InitPerRun();
  // stop init timer
  auto end_0 = std::chrono::high_resolution_clock::now();
  // start compute timer
  s_compute = std::chrono::high_resolution_clock::now();
  ForwardSub(context, a, b, m, size, timing);

#ifdef DEBUG
  printf("The result of matrix m is: \n");
  PrintMat(m, size, size, size);
  printf("The result of matrix a is: \n");
  PrintMat(a, size, size, size);
  printf("The result of array b is: \n");
  PrintAry(b, size);
#endif
  BackSub();

#ifdef DEBUG
  printf("The final solution is: \n");
  PrintAry(finalVec, size);
#endif
  // stop compute timer
  e_compute = std::chrono::high_resolution_clock::now();

  free(m);
  free(a);
  free(b);
  free(finalVec);

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

  cl_freeKernel(fan1_kernel);
  cl_freeKernel(fan2_kernel);
  cl_freeProgram(gaussianElim_program);
  cl_cleanup();

  return 0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(cl_context context, float *a, float *b, float *m, int size,
                int timing) {
  // 2. set up memory on device and send ipts data to device

  cl_mem a_dev, b_dev, m_dev;

  cl_int error = 0;

  a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
                         sizeof(float) * size * size, NULL, &error);

  b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL,
                         &error);

  m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
                         sizeof(float) * size * size, NULL, &error);

  cl_command_queue command_queue = cl_getCommandQueue();

  error = clEnqueueWriteBuffer(command_queue, a_dev,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(float) * size * size, a, 0, NULL, NULL);

  error = clEnqueueWriteBuffer(command_queue, b_dev,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(float) * size, b, 0, NULL, NULL);

  error = clEnqueueWriteBuffer(command_queue, m_dev,
                               1, // change to 0 for nonblocking write
                               0, // offset
                               sizeof(float) * size * size, m, 0, NULL, NULL);

  // 3. Determine block sizes
  size_t globalWorksizeFan1[1];
  size_t globalWorksizeFan2[2];
  size_t localWorksizeFan1Buf[1] = {BLOCK_SIZE_0};
  size_t localWorksizeFan2Buf[2] = {BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
  size_t *localWorksizeFan1 = NULL;
  size_t *localWorksizeFan2 = NULL;

  globalWorksizeFan1[0] = size;
  globalWorksizeFan2[0] = size;
  globalWorksizeFan2[1] = size;

  if (localWorksizeFan1Buf[0]) {
    localWorksizeFan1 = localWorksizeFan1Buf;
    globalWorksizeFan1[0] =
        (int)ceil(globalWorksizeFan1[0] / (double)localWorksizeFan1Buf[0]) *
        localWorksizeFan1Buf[0];
  }
  if (localWorksizeFan2Buf[0]) {
    localWorksizeFan2 = localWorksizeFan2Buf;
    globalWorksizeFan2[0] =
        (int)ceil(globalWorksizeFan2[0] / (double)localWorksizeFan2Buf[0]) *
        localWorksizeFan2Buf[0];
    globalWorksizeFan2[1] =
        (int)ceil(globalWorksizeFan2[1] / (double)localWorksizeFan2Buf[1]) *
        localWorksizeFan2Buf[1];
  }

  int t;
  // 4. Setup and Run kernels
  for (t = 0; t < (size - 1); t++) {
    // kernel args
    cl_int argchk;
    argchk = clSetKernelArg(fan1_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
    argchk |= clSetKernelArg(fan1_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
    argchk |= clSetKernelArg(fan1_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
    argchk |= clSetKernelArg(fan1_kernel, 3, sizeof(int), (void *)&size);
    argchk |= clSetKernelArg(fan1_kernel, 4, sizeof(int), (void *)&t);

#ifdef DEBUG
    cl_errChk(argchk, "ERROR in Setting Fan1 kernel args", true);
#endif
    // launch kernel
    error = clEnqueueNDRangeKernel(command_queue, fan1_kernel, 1, 0,
                                   globalWorksizeFan1, localWorksizeFan1, 0,
                                   NULL, NULL);

#ifdef DEBUG
    cl_errChk(error, "ERROR in Executing Fan1 Kernel", true);
#endif
    // kernel args
    argchk = clSetKernelArg(fan2_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
    argchk |= clSetKernelArg(fan2_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
    argchk |= clSetKernelArg(fan2_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
    argchk |= clSetKernelArg(fan2_kernel, 3, sizeof(int), (void *)&size);
    argchk |= clSetKernelArg(fan2_kernel, 4, sizeof(int), (void *)&t);

#ifdef DEBUG
    cl_errChk(argchk, "ERROR in Setting Fan2 kernel args", true);
#endif
    // launch kernel
    error = clEnqueueNDRangeKernel(command_queue, fan2_kernel, 2, 0,
                                   globalWorksizeFan2, NULL, 0, NULL, NULL);

#ifdef DEBUG
    cl_errChk(error, "ERROR in Executing Fan1 Kernel", true);
#endif
  }
  // 5. transfer data off of device
  error = clEnqueueReadBuffer(command_queue, a_dev,
                              1, // change to 0 for nonblocking write
                              0, // offset
                              sizeof(float) * size * size, a, 0, NULL, NULL);

#ifdef DEBUG
  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
#endif
  error = clEnqueueReadBuffer(command_queue, b_dev,
                              1, // change to 0 for nonblocking write
                              0, // offset
                              sizeof(float) * size, b, 0, NULL, NULL);
#ifdef DEBUG
  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
#endif
  error = clEnqueueReadBuffer(command_queue, m_dev,
                              1, // change to 0 for nonblocking write
                              0, // offset
                              sizeof(float) * size * size, m, 0, NULL, NULL);
#ifdef DEBUG
  cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
#endif
  cl_freeMem(a_dev);
  cl_freeMem(b_dev);
  cl_freeMem(m_dev);
}

// Ke Wang add a function to generate input internally
int parseCommandline(int argc, char *argv[], char *filename, int *q, int *t,
                     int *p, int *d, int *size) {
  int i;
  if (argc < 2)
    return 1; // error
  // strncpy(filename,argv[1],100);
  char flag;

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 's': // platform
        i++;
        *size = atoi(argv[i]);
        break;
      case 'f': // platform
        i++;
        strncpy(filename, argv[i], 100);
        break;
      case 'h': // help
        return 1;
        break;
      case 'q': // quiet
        *q = 1;
        break;
      case 't': // timing
        *t = 1;
        break;
      case 'p': // platform
        i++;
        *p = atoi(argv[i]);
        break;
      case 'd': // device
        i++;
        *d = atoi(argv[i]);
        break;
      }
    }
  }
  if ((*d >= 0 && *p < 0) ||
      (*p >= 0 &&
       *d < 0)) // both p and d must be specified if either are specified
    return 1;
  return 0;
}

void printUsage() {
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination [filename] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination matrix4.txt\n");
  printf("\n");
  printf("filename     the filename that holds the matrix data\n");
  printf("\n");
  printf("-h           Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and "
         "device)\n");
  printf("-d [int]     Choose the device (must choose both platform and "
         "device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() {
  int i;
  for (i = 0; i < size * size; i++)
    *(m + i) = 0.0;
}
void BackSub() {
  // solve "bottom up"
  int i, j;
  for (i = 0; i < size; i++) {
    finalVec[size - i - 1] = b[size - i - 1];
    for (j = 0; j < i; j++) {
      finalVec[size - i - 1] -= *(a + size * (size - i - 1) + (size - j - 1)) *
                                finalVec[size - j - 1];
    }
    finalVec[size - i - 1] =
        finalVec[size - i - 1] / *(a + size * (size - i - 1) + (size - i - 1));
  }
}
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      fscanf(fp, "%f", ary + size * i + j);
    }
  }
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size) {
  int i;

  for (i = 0; i < ary_size; i++) {
    fscanf(fp, "%f", &ary[i]);
  }
}
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol) {
  int i, j;

  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      printf("%8.2f ", *(ary + size * i + j));
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size) {
  int i;
  for (i = 0; i < ary_size; i++) {
    printf("%.2e ", ary[i]);
  }
  printf("\n\n");
}
#endif
