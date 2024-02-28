#include "CLHelper.h"
#include <CL/cl.h>
#include <chrono>
#include <fcntl.h>
#include <float.h>
#include <iostream>>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
//#define DEBUG
#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932

int threads_per_block = 256;
std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;

//#define BREAKDOWNS
#ifdef BREAKDOWNS
std::chrono::high_resolution_clock::time_point s_b0;
std::chrono::high_resolution_clock::time_point e_b0;
std::chrono::high_resolution_clock::time_point s_b1;
std::chrono::high_resolution_clock::time_point e_b1;
std::chrono::high_resolution_clock::time_point s_b2;
std::chrono::high_resolution_clock::time_point e_b2;
#endif

/***** kernel variables ******/
cl_kernel kernel_likelihood;
cl_kernel kernel_sum;
cl_kernel kernel_normalize_weights;
cl_kernel kernel_find_index;

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
 */
long M = INT_MAX;
/**
@var A value for LCG
 */
int A = 1103515245;
/**
@var C value for LCG
 */
int C = 12345;

//#include "oclUtils.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

void ocl_print_double_array(cl_command_queue cmd_q, cl_mem array_GPU,
                            size_t size) {
  // allocate temporary array for printing
  double *mem = (double *)calloc(size, sizeof(double));

  // transfer data from device
  cl_int err = clEnqueueReadBuffer(cmd_q, array_GPU, 1, 0,
                                   sizeof(double) * size, mem, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: Memcopy Out\n");
    return;
  }

  printf("PRINTING ARRAY VALUES\n");
  // print values in memory
  for (size_t i = 0; i < size; ++i) {
    printf("[%d]:%0.6f\n", i, mem[i]);
  }
  printf("FINISHED PRINTING ARRAY VALUES\n");

  // clean up memory
  free(mem);
  mem = NULL;
}

/**
 * Generates a uniformly distributed random number using the provided seed and
 * GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */
double randu(int *seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((double)M));
}

/**
 * Generates a normally distributed random number using the Box-Muller
 * transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a double representing random number generated using the Box-Muller
 * algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing
 * value for normal random distribution
 */
double randn(int *seed, int index) {
  /*Box-Muller algorithm*/
  double u = randu(seed, index);
  double v = randu(seed, index);
  double cosine = cos(2 * PI * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

/**
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value
 * > input value
 */
double roundDouble(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the
 * testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, unsigned char *array3D, int *dimX,
           int *dimY, int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal
 * distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char *array3D, int *dimX, int *dimY, int *dimZ,
              int *seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] =
            array3D[x * *dimY * *dimZ + y * *dimZ + z] +
            (unsigned char)(5 * randn(seed, 0));
      }
    }
  }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      double distance = sqrt(pow((double)(x - radius + 1), 2) +
                             pow((double)(y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
    }
  }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(unsigned char *matrix, int posX, int posY, int posZ,
                   int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      double distance =
          sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(unsigned char *matrix, int dimX, int dimY, int dimZ,
                   int error, unsigned char *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int *se, int numOnes, int *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(unsigned char *I, int IszX, int IszY, int Nfr, int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = (int)roundDouble(IszY / 2.0);
  int y0 = (int)roundDouble(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  unsigned char *newMatrix =
      (unsigned char *)calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);
  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}

/**
 * Finds the first element in the CDF that is greater than or equal to the
 * provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the
 * last index
 */
int findIndex(double *CDF, int lengthCDF, double value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In
 * addition, it references a provided MATLAB function which takes the video, the
 * objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
int particleFilter(unsigned char *I, int IszX, int IszY, int Nfr, int *seed,
                   int Nparticles) {
  int max_size = IszX * IszY * Nfr;
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)calloc(diameter * diameter, sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int *objxy = (int *)calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);
  // initial weights are all equal (1/Nparticles)
  double *weights = (double *)calloc(Nparticles, sizeof(double));
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((double)(Nparticles));
  }
  char *s_likelihood_kernel = "likelihood_kernel";
  char *s_sum_kernel = "sum_kernel";
  char *s_normalize_weights_kernel = "normalize_weights_kernel";
  char *s_find_index_kernel = "find_index_kernel";

  // initial likelihood to 0.0
  double *likelihood = (double *)calloc(Nparticles, sizeof(double));
  double *arrayX = (double *)calloc(Nparticles, sizeof(double));
  double *arrayY = (double *)calloc(Nparticles, sizeof(double));
  double *xj = (double *)calloc(Nparticles, sizeof(double));
  double *yj = (double *)calloc(Nparticles, sizeof(double));
  double *CDF = (double *)calloc(Nparticles, sizeof(double));
  int *ind = (int *)calloc(countOnes * Nparticles, sizeof(int));
  double *u = (double *)calloc(Nparticles, sizeof(double));

#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  // GPU copies of arrays
  cl_mem arrayX_GPU;
  cl_mem arrayY_GPU;
  cl_mem xj_GPU;
  cl_mem yj_GPU;
  cl_mem CDF_GPU;
  cl_mem likelihood_GPU;
  cl_mem I_GPU;
  cl_mem weights_GPU;
  cl_mem objxy_GPU;

  cl_mem u_GPU;
  cl_mem ind_GPU;
  cl_mem seed_GPU;
  cl_mem partial_sums;

  // OpenCL memory allocation
  cl_int err;

  arrayX_GPU = _clMallocRW(sizeof(double) * Nparticles, arrayX);
  arrayY_GPU = _clMallocRW(sizeof(double) * Nparticles, arrayY);
  xj_GPU = _clMallocRW(sizeof(double) * Nparticles, xj);
  yj_GPU = _clMallocRW(sizeof(double) * Nparticles, yj);
  CDF_GPU = _clMallocRW(sizeof(double) * Nparticles, CDF);
  u_GPU = _clMallocRW(sizeof(double) * Nparticles, u);
  likelihood_GPU = _clMallocRW(sizeof(double) * Nparticles, likelihood);
  weights_GPU = _clMallocRW(sizeof(double) * Nparticles, weights);
  I_GPU = _clMallocRW(sizeof(unsigned char) * IszX * IszY * Nfr, I);
  objxy_GPU = _clMallocRW(2 * sizeof(int) * countOnes, objxy);
  ind_GPU = _clMallocRW(sizeof(int) * countOnes * Nparticles, ind);
  seed_GPU = _clMallocRW(sizeof(int) * Nparticles, seed);
  partial_sums = _clMallocRW(sizeof(double) * Nparticles, likelihood);

#ifdef BREAKDOWNS
  clFinish(oclHandles.queue);
  e_b0 = std::chrono::high_resolution_clock::now();
#endif

  // Donnie - this loop is different because in this kernel, arrayX and arrayY
  //  are set equal to xj before every iteration, so effectively, arrayX and
  //  arrayY will be set to xe and ye before the first iteration.
  for (x = 0; x < Nparticles; x++) {
    xj[x] = xe;
    yj[x] = ye;
  }

  int k;
  // double * Ik = (double *)calloc(IszX*IszY, sizeof(double));
  int indX, indY;
  // start send
#ifdef BREAKDOWNS
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  // OpenCL memory copy
  err = clEnqueueWriteBuffer(oclHandles.queue, I_GPU, 1, 0,
                             sizeof(unsigned char) * IszX * IszY * Nfr, I, 0, 0,
                             0);
  if (err != CL_SUCCESS) {
    printf("ERROR: clEnqueueWriteBuffer I_GPU (size:%d) => %d\n",
           IszX * IszY * Nfr, err);
    return -1;
  }
  err = clEnqueueWriteBuffer(oclHandles.queue, objxy_GPU, 1, 0,
                             2 * sizeof(int) * countOnes, objxy, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: clEnqueueWriteBuffer objxy_GPU (size:%d) => %d\n", countOnes,
           err);
    return -1;
  }
  err = clEnqueueWriteBuffer(oclHandles.queue, weights_GPU, 1, 0,
                             sizeof(double) * Nparticles, weights, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: clEnqueueWriteBuffer weights_GPU (size:%d) => %d\n",
           Nparticles, err);
    return -1;
  }
  err = clEnqueueWriteBuffer(oclHandles.queue, xj_GPU, 1, 0,
                             sizeof(double) * Nparticles, xj, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: clEnqueueWriteBuffer arrayX_GPU (size:%d) => %d\n",
           Nparticles, err);
    return -1;
  }
  err = clEnqueueWriteBuffer(oclHandles.queue, yj_GPU, 1, 0,
                             sizeof(double) * Nparticles, yj, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: clEnqueueWriteBuffer arrayY_GPU (size:%d) => %d\n",
           Nparticles, err);
    return -1;
  }
  err = clEnqueueWriteBuffer(oclHandles.queue, seed_GPU, 1, 0,
                             sizeof(int) * Nparticles, seed, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: clEnqueueWriteBuffer seed_GPU (size:%d) => %d\n", Nparticles,
           err);
    return -1;
  }
#ifdef BREAKDOWNS
  clFinish(oclHandles.queue);
  e_b1 = std::chrono::high_resolution_clock::now();
#endif

  size_t global_work[3] = {256, 1, 1};

  // size_t local_work[3] = {threads_per_block, 1, 1};
  size_t local_work[3] = {64, 1, 1};

#ifdef BREAKDOWNS
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  for (k = 1; k < Nfr; k++) {
    clSetKernelArg(oclHandles.kernel[0], 0, sizeof(void *),
                   (void *)&arrayX_GPU);
    clSetKernelArg(oclHandles.kernel[0], 1, sizeof(void *),
                   (void *)&arrayY_GPU);
    clSetKernelArg(oclHandles.kernel[0], 2, sizeof(void *), (void *)&xj_GPU);
    clSetKernelArg(oclHandles.kernel[0], 3, sizeof(void *), (void *)&yj_GPU);
    clSetKernelArg(oclHandles.kernel[0], 4, sizeof(void *), (void *)&CDF_GPU);
    clSetKernelArg(oclHandles.kernel[0], 5, sizeof(void *), (void *)&ind_GPU);
    clSetKernelArg(oclHandles.kernel[0], 6, sizeof(void *), (void *)&objxy_GPU);
    clSetKernelArg(oclHandles.kernel[0], 7, sizeof(void *),
                   (void *)&likelihood_GPU);
    clSetKernelArg(oclHandles.kernel[0], 8, sizeof(void *), (void *)&I_GPU);
    clSetKernelArg(oclHandles.kernel[0], 9, sizeof(void *), (void *)&u_GPU);
    clSetKernelArg(oclHandles.kernel[0], 10, sizeof(void *),
                   (void *)&weights_GPU);
    clSetKernelArg(oclHandles.kernel[0], 11, sizeof(cl_int),
                   (void *)&Nparticles);
    clSetKernelArg(oclHandles.kernel[0], 12, sizeof(cl_int),
                   (void *)&countOnes);
    clSetKernelArg(oclHandles.kernel[0], 13, sizeof(cl_int), (void *)&max_size);
    clSetKernelArg(oclHandles.kernel[0], 14, sizeof(cl_int), (void *)&k);
    clSetKernelArg(oclHandles.kernel[0], 15, sizeof(cl_int), (void *)&IszY);
    clSetKernelArg(oclHandles.kernel[0], 16, sizeof(cl_int), (void *)&Nfr);
    clSetKernelArg(oclHandles.kernel[0], 17, sizeof(void *), (void *)&seed_GPU);
    clSetKernelArg(oclHandles.kernel[0], 18, sizeof(void *),
                   (void *)&partial_sums);
    clSetKernelArg(oclHandles.kernel[0], 19, threads_per_block * sizeof(double),
                   NULL);

    err = clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[0], 1,
                                 NULL, global_work, local_work, 0, 0, 0);
    if (err != CL_SUCCESS) {
      printf("ERROR: clEnqueueNDRangeKernel(kernel_likelihood)=>%d failed\n",
             err);
      return -1;
    }
    clSetKernelArg(oclHandles.kernel[1], 0, sizeof(void *),
                   (void *)&partial_sums);
    clSetKernelArg(oclHandles.kernel[1], 1, sizeof(cl_int),
                   (void *)&Nparticles);

    err = clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[1], 1,
                                 NULL, global_work, local_work, 0, 0, 0);
    if (err != CL_SUCCESS) {
      printf("ERROR: clEnqueueNDRangeKernel(kernel_sum)=>%d failed\n", err);
      return -1;
    }

    clSetKernelArg(oclHandles.kernel[2], 0, sizeof(void *),
                   (void *)&weights_GPU);
    clSetKernelArg(oclHandles.kernel[2], 1, sizeof(cl_int),
                   (void *)&Nparticles);
    clSetKernelArg(oclHandles.kernel[2], 2, sizeof(void *),
                   (void *)&partial_sums); //*/
    clSetKernelArg(oclHandles.kernel[2], 3, sizeof(void *), (void *)&CDF_GPU);
    clSetKernelArg(oclHandles.kernel[2], 4, sizeof(void *), (void *)&u_GPU);
    clSetKernelArg(oclHandles.kernel[2], 5, sizeof(void *), (void *)&seed_GPU);

    err = clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[2], 1,
                                 NULL, global_work, local_work, 0, 0, 0);
    if (err != CL_SUCCESS) {
      printf("ERROR: clEnqueueNDRangeKernel(normalize_weights)=>%d failed\n",
             err);
      return -1;
    }

    clSetKernelArg(oclHandles.kernel[3], 0, sizeof(void *),
                   (void *)&arrayX_GPU);
    clSetKernelArg(oclHandles.kernel[3], 1, sizeof(void *),
                   (void *)&arrayY_GPU);
    clSetKernelArg(oclHandles.kernel[3], 2, sizeof(void *), (void *)&CDF_GPU);
    clSetKernelArg(oclHandles.kernel[3], 3, sizeof(void *), (void *)&u_GPU);
    clSetKernelArg(oclHandles.kernel[3], 4, sizeof(void *), (void *)&xj_GPU);
    clSetKernelArg(oclHandles.kernel[3], 5, sizeof(void *), (void *)&yj_GPU);
    clSetKernelArg(oclHandles.kernel[3], 6, sizeof(void *),
                   (void *)&weights_GPU);
    clSetKernelArg(oclHandles.kernel[3], 7, sizeof(cl_int),
                   (void *)&Nparticles);
    // KERNEL FUNCTION CALL
    err = clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[3], 1,
                                 NULL, global_work, local_work, 0, 0, 0);
    if (err != CL_SUCCESS) {
      printf("ERROR: clEnqueueNDRangeKernel(find_index)=>%d failed\n", err);
      return -1;
    }

  } // end loop

#ifdef BREAKDOWNS
  clFinish(oclHandles.queue);
  e_b2 = std::chrono::high_resolution_clock::now();
#endif

  // OpenCL memory copying back from GPU to CPU memory
  err = clEnqueueReadBuffer(oclHandles.queue, arrayX_GPU, 1, 0,
                            sizeof(double) * Nparticles, arrayX, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: 1. Memcopy Out\n");
    return -1;
  }
  err = clEnqueueReadBuffer(oclHandles.queue, arrayY_GPU, 1, 0,
                            sizeof(double) * Nparticles, arrayY, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: 2. Memcopy Out\n");
    return -1;
  }
  err = clEnqueueReadBuffer(oclHandles.queue, weights_GPU, 1, 0,
                            sizeof(double) * Nparticles, weights, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("ERROR: 3. Memcopy Out\n");
    return -1;
  }
#ifdef DEBUG
  xe = 0;
  ye = 0;
  // estimate the object location by expected values
  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  double distance = sqrt(pow((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                         pow((double)(ye - (int)roundDouble(IszX / 2.0)), 2));

  // Output results
  FILE *fid;
  fid = fopen("output.txt", "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return -1;
  }
  fprintf(fid, "XE: %lf\n", xe);
  fprintf(fid, "YE: %lf\n", ye);
  fprintf(fid, "distance: %lf\n", distance);
  fclose(fid);
#endif
  // OpenCL freeing of memory
  clReleaseMemObject(weights_GPU);
  clReleaseMemObject(arrayY_GPU);
  clReleaseMemObject(arrayX_GPU);

  // free regular memory
  free(likelihood);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(ind);
  free(u);
}

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  char *usage = "double.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  // check number of arguments
  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  // check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") ||
      strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  // converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }
  // establish seed
  int *seed = (int *)malloc(sizeof(int) * Nparticles);
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = time(0) * i;

  unsigned char *I =
      (unsigned char *)calloc(IszX * IszY * Nfr, sizeof(unsigned char));

  s_init_fpga_timer = std::chrono::high_resolution_clock::now();
  _clInit();
  e_init_fpga_timer = std::chrono::high_resolution_clock::now();
  auto end_0 = std::chrono::high_resolution_clock::now();
  s_compute = std::chrono::high_resolution_clock::now();

  // call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);

  // call particle filter
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);

  e_compute = std::chrono::high_resolution_clock::now();
  free(seed);
  free(I);
  auto end = std::chrono::high_resolution_clock::now();

#ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown Computation #####" << std::endl;
  std::chrono::duration<double, std::milli> allocation = e_b0 - s_b0;
  std::cerr << "Malloc time: " << allocation.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer = e_b1 - s_b1;
  std::cerr << "Mmcpy time: " << transfer.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> compute = e_b2 - s_b2;
  std::cerr << "Compute time: " << compute.count() << " ms" << std::endl;

  std::cerr << " #################################" << std::endl;
#endif

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

  _clRelease();
  return 0;
}
