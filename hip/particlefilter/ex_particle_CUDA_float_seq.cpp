#include "hip/hip_runtime.h"
#include <chrono>
#include <fcntl.h>
#include <float.h>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932

const int threads_per_block = 512;
#define WARMUP
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;

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

/*****************************
 * CHECK_ERROR
 * Checks for CUDA errors and prints them to the screen to help with
 * debugging of CUDA related programming
 *****************************/
void check_error(hipError_t e) {
  if (e != hipSuccess) {
    printf("\nCUDA error: %s\n", hipGetErrorString(e));
    exit(1);
  }
}

void cuda_print_double_array(double *array_GPU, size_t size) {
  // allocate temporary array for printing
  double *mem = (double *)malloc(sizeof(double) * size);

  // transfer data from device
  hipMemcpy(mem, array_GPU, sizeof(double) * size, hipMemcpyDeviceToHost);

  printf("PRINTING ARRAY VALUES\n");
  // print values in memory
  for (size_t i = 0; i < size; ++i) {
    printf("[%zu]:%0.6f\n", i, mem[i]);
  }
  printf("FINISHED PRINTING ARRAY VALUES\n");

  // clean up memory
  free(mem);
  mem = NULL;
}

/********************************
 * CALC LIKELIHOOD SUM
 * DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 -
 *(IK[IND] - 228)^2)/ 100 param 1 I 3D matrix param 2 current ind array param 3
 *length of ind array returns a double representing the sum
 ********************************/
__device__ double calcLikelihoodSum(unsigned char *I, int *ind, int numOnes,
                                    int index) {
  double likelihoodSum = 0.0;
  int x;
  for (x = 0; x < numOnes; x++)
    likelihoodSum += (pow((double)(I[ind[index * numOnes + x]] - 100), 2) -
                      pow((double)(I[ind[index * numOnes + x]] - 228), 2)) /
                     50.0;
  return likelihoodSum;
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
 *****************************/
__device__ void cdfCalc(double *CDF, double *weights, int Nparticles) {
  int x;
  CDF[0] = weights[0];
  for (x = 1; x < Nparticles; x++) {
    CDF[x] = weights[x] + CDF[x - 1];
  }
}

/*****************************
 * RANDU
 * GENERATES A UNIFORM DISTRIBUTION
 * returns a double representing a randomily generated number from a uniform
 *distribution with range [0, 1)
 ******************************/
__device__ double d_randu(int *seed, int index) {

  int M = INT_MAX;
  int A = 1103515245;
  int C = 12345;
  int num = A * seed[index] + C;
  seed[index] = num % M;

  return fabs(seed[index] / ((double)M));
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

double test_randn(int *seed, int index) {
  // Box-Muller algortihm
  double pi = 3.14159265358979323846;
  double u = randu(seed, index);
  double v = randu(seed, index);
  double cosine = cos(2 * pi * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

__device__ double d_randn(int *seed, int index) {
  // Box-Muller algortihm
  double pi = 3.14159265358979323846;
  double u = d_randu(seed, index);
  double v = d_randu(seed, index);
  double cosine = cos(2 * pi * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparcitles
 ****************************/
__device__ double updateWeights(double *weights, double *likelihood,
                                int Nparticles) {
  int x;
  double sum = 0;
  for (x = 0; x < Nparticles; x++) {
    weights[x] = weights[x] * exp(likelihood[x]);
    sum += weights[x];
  }
  return sum;
}

__device__ int findIndexBin(double *CDF, int beginIndex, int endIndex,
                            double value) {
  if (endIndex < beginIndex)
    return -1;
  int middleIndex;
  while (endIndex > beginIndex) {
    middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
    if (CDF[middleIndex] >= value) {
      if (middleIndex == 0)
        return middleIndex;
      else if (CDF[middleIndex - 1] < value)
        return middleIndex;
      else if (CDF[middleIndex - 1] == value) {
        while (CDF[middleIndex] == value && middleIndex >= 0)
          middleIndex--;
        middleIndex++;
        return middleIndex;
      }
    }
    if (CDF[middleIndex] > value)
      endIndex = middleIndex - 1;
    else
      beginIndex = middleIndex + 1;
  }
  return -1;
}

/** added this function. was missing in original double version.
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value
 * > input value
 */
__device__ double dev_round_double(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5f)
    return newValue;
  else
    return newValue++;
}

/*****************************
 * CUDA Find Index Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param3: CDF
 * param4: u
 * param5: xj
 * param6: yj
 * param7: weights
 * param8: Nparticles
 *****************************/
__global__ void find_index_kernel(double *arrayX, double *arrayY, double *CDF,
                                  double *u, double *xj, double *yj,
                                  double *weights, int Nparticles) {

  int block_id = hipBlockIdx_x;
  int i = hipBlockDim_x * block_id + hipThreadIdx_x;

  if (i < Nparticles) {

    int index = -1;
    int x;

    for (x = 0; x < Nparticles; x++) {
      if (CDF[x] >= u[i]) {
        index = x;
        break;
      }
    }
    if (index == -1) {
      index = Nparticles - 1;
    }

    xj[i] = arrayX[index];
    yj[i] = arrayY[index];

    // weights[i] = 1 / ((double) (Nparticles)); //moved this code to the
    // beginning of likelihood kernel
  }

  __syncthreads(); // idx.barrier.wait();
}

__global__ void normalize_weights_kernel(double *weights, int Nparticles,
                                         double *partial_sums, double *CDF,
                                         double *u, int *seed) {

  int block_id = hipBlockIdx_x;
  int i = hipBlockDim_x * block_id + hipThreadIdx_x;
  __shared__ double u1, sumWeights;

  if (0 == hipThreadIdx_x)
    sumWeights = partial_sums[0];

  __syncthreads(); // idx.barrier.wait();

  if (i < Nparticles) {
    weights[i] = weights[i] / sumWeights;
  }

  __syncthreads(); // idx.barrier.wait();

  if (i == 0) {
    cdfCalc(CDF, weights, Nparticles);
    u[0] =
        (1 / ((double)(Nparticles))) *
        d_randu(
            seed,
            i); // do this to allow all threads in all blocks to use the same u1
  }

  __syncthreads(); // idx.barrier.wait();

  if (0 == hipThreadIdx_x)
    u1 = u[0];

  __syncthreads(); // idx.barrier.wait();

  if (i < Nparticles) {
    u[i] = u1 + i / ((double)(Nparticles));
  }
}

__global__ void sum_kernel(double *partial_sums, int Nparticles) {

  int block_id = hipBlockIdx_x;
  int i = hipBlockDim_x * block_id + hipThreadIdx_x;

  if (i == 0) {
    int x;
    double sum = 0.0;
    int num_blocks = ceil((double)Nparticles / (double)threads_per_block);
    for (x = 0; x < num_blocks; x++) {
      sum += partial_sums[x];
    }
    partial_sums[0] = sum;
  }
}

/*****************************
 * CUDA Likelihood Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param2.5: CDF
 * param3: ind
 * param4: objxy
 * param5: likelihood
 * param6: I
 * param6.5: u
 * param6.75: weights
 * param7: Nparticles
 * param8: countOnes
 * param9: max_size
 * param10: k
 * param11: IszY
 * param12: Nfr
 *****************************/
__global__ void likelihood_kernel(double *arrayX, double *arrayY, double *xj,
                                  double *yj, double *CDF, int *ind, int *objxy,
                                  double *likelihood, unsigned char *I,
                                  double *u, double *weights, int Nparticles,
                                  int countOnes, int max_size, int k, int IszY,
                                  int Nfr, int *seed, double *partial_sums) {

  int block_id = hipBlockIdx_x;
  int i = hipBlockDim_x * block_id + hipThreadIdx_x;
  int y;

  int indX, indY;
  __shared__ double buffer[512];
  if (i < Nparticles) {
    arrayX[i] = xj[i];
    arrayY[i] = yj[i];

    weights[i] =
        1 / ((double)(Nparticles)); // Donnie - moved this line from end of
                                    // find_index_kernel to prevent all weights
                                    // from being reset before calculating
                                    // position on final iteration.

    arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i);
    arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i);
  }

  __syncthreads(); // idx.barrier.wait();

  if (i < Nparticles) {
    for (y = 0; y < countOnes; y++) {
      // added dev_round_double() to be consistent with roundDouble
      indX = dev_round_double(arrayX[i]) + objxy[y * 2 + 1];
      indY = dev_round_double(arrayY[i]) + objxy[y * 2];

      ind[i * countOnes + y] =
          (int)fabs((float)(indX * IszY * Nfr + indY * Nfr + k));
      if (ind[i * countOnes + y] >= max_size)
        ind[i * countOnes + y] = 0;
    }
    likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

    likelihood[i] = likelihood[i] / countOnes;

    weights[i] =
        weights[i] * exp(likelihood[i]); // Donnie Newell - added the missing
                                         // exponential function call
  }

  buffer[hipThreadIdx_x] = 0.0;

  __syncthreads(); // idx.barrier.wait();

  if (i < Nparticles) {

    buffer[hipThreadIdx_x] = weights[i];
  }

  __syncthreads(); // idx.barrier.wait();

  // this doesn't account for the last block that isn't full
  for (unsigned int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
    if (hipThreadIdx_x < s) {
      buffer[hipThreadIdx_x] += buffer[hipThreadIdx_x + s];
    }

    __syncthreads(); // idx.barrier.wait();
  }
  if (hipThreadIdx_x == 0) {
    partial_sums[hipBlockIdx_x] = buffer[0];
  }

  __syncthreads(); // idx.barrier.wait();
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
      if (se[x * diameter + y] == 1) {
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
 * the foreground intensity and the background intensity is known
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
    xk = (int)fabs((float)(x0 + (k - 1)));
    yk = (int)fabs((float)(y0 - 2 * (k - 1)));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  unsigned char *newMatrix =
      (unsigned char *)malloc(sizeof(unsigned char) * IszX * IszY * Nfr);
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
void particleFilter(unsigned char *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles) {
  int max_size = IszX * IszY * Nfr;
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)malloc(diameter * diameter * sizeof(int));
  memset(disk, 0, diameter * diameter * sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int *objxy = (int *)malloc(countOnes * 2 * sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);
  // initial weights are all equal (1/Nparticles)
  double *weights = (double *)malloc(sizeof(double) * Nparticles);
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((double)(Nparticles));
  }

  // initial likelihood to 0.0
  double *likelihood = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayX = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayY = (double *)malloc(sizeof(double) * Nparticles);
  double *xj = (double *)malloc(sizeof(double) * Nparticles);
  double *yj = (double *)malloc(sizeof(double) * Nparticles);
  double *CDF = (double *)malloc(sizeof(double) * Nparticles);

  // GPU copies of arrays
  double *arrayX_GPU;
  double *arrayY_GPU;
  double *xj_GPU;
  double *yj_GPU;
  double *CDF_GPU;
  double *likelihood_GPU;
  unsigned char *I_GPU;
  double *weights_GPU;
  int *objxy_GPU;

  int *ind = (int *)malloc(sizeof(int) * countOnes * Nparticles);
  int *ind_GPU;
  double *u = (double *)malloc(sizeof(double) * Nparticles);
  double *u_GPU;
  int *seed_GPU;
  double *partial_sums;

  // CUDA memory allocation
  check_error(hipMalloc((void **)&arrayX_GPU, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&arrayY_GPU, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&xj_GPU, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&yj_GPU, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&CDF_GPU, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&u_GPU, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&likelihood_GPU, sizeof(double) * Nparticles));
  // set likelihood to zero
  check_error(
      hipMemset((void *)likelihood_GPU, 0, sizeof(double) * Nparticles));
  check_error(hipMalloc((void **)&weights_GPU, sizeof(double) * Nparticles));
  check_error(
      hipMalloc((void **)&I_GPU, sizeof(unsigned char) * IszX * IszY * Nfr));
  check_error(hipMalloc((void **)&objxy_GPU, sizeof(int) * 2 * countOnes));
  check_error(
      hipMalloc((void **)&ind_GPU, sizeof(int) * countOnes * Nparticles));
  check_error(hipMalloc((void **)&seed_GPU, sizeof(int) * Nparticles));
  check_error(hipMalloc((void **)&partial_sums, sizeof(double) * Nparticles));

  // Donnie - this loop is different because in this kernel, arrayX and arrayY
  //   are set equal to xj before every iteration, so effectively, arrayX and
  //   arrayY will be set to xe and ye before the first iteration.
  for (x = 0; x < Nparticles; x++) {

    xj[x] = xe;
    yj[x] = ye;
  }

  int k;
  int indX, indY;
  // start send
  check_error(hipMemcpy(I_GPU, I, sizeof(unsigned char) * IszX * IszY * Nfr,
                        hipMemcpyHostToDevice));
  check_error(hipMemcpy(objxy_GPU, objxy, sizeof(int) * 2 * countOnes,
                        hipMemcpyHostToDevice));
  check_error(hipMemcpy(weights_GPU, weights, sizeof(double) * Nparticles,
                        hipMemcpyHostToDevice));
  check_error(hipMemcpy(xj_GPU, xj, sizeof(double) * Nparticles,
                        hipMemcpyHostToDevice));
  check_error(hipMemcpy(yj_GPU, yj, sizeof(double) * Nparticles,
                        hipMemcpyHostToDevice));
  check_error(hipMemcpy(seed_GPU, seed, sizeof(int) * Nparticles,
                        hipMemcpyHostToDevice));
  int num_blocks = ceil((double)Nparticles / (double)threads_per_block);

  for (k = 1; k < Nfr; k++) {

    hipLaunchKernelGGL(
        likelihood_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
        arrayX_GPU, arrayY_GPU, xj_GPU, yj_GPU, CDF_GPU, ind_GPU, objxy_GPU,
        likelihood_GPU, I_GPU, u_GPU, weights_GPU, Nparticles, countOnes,
        max_size, k, IszY, Nfr, seed_GPU, partial_sums);

    hipLaunchKernelGGL(sum_kernel, dim3(num_blocks), dim3(threads_per_block), 0,
                       0, partial_sums, Nparticles);

    hipLaunchKernelGGL(normalize_weights_kernel, dim3(num_blocks),
                       dim3(threads_per_block), 0, 0, weights_GPU, Nparticles,
                       partial_sums, CDF_GPU, u_GPU, seed_GPU);

    hipLaunchKernelGGL(find_index_kernel, dim3(num_blocks),
                       dim3(threads_per_block), 0, 0, arrayX_GPU, arrayY_GPU,
                       CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU, Nparticles);

  } // end loop

  hipFree(xj_GPU);
  hipFree(yj_GPU);
  hipFree(CDF_GPU);
  hipFree(u_GPU);
  hipFree(likelihood_GPU);
  hipFree(I_GPU);
  hipFree(objxy_GPU);
  hipFree(ind_GPU);
  hipFree(seed_GPU);
  hipFree(partial_sums);

#ifdef OUTPUT
  xe = 0;
  ye = 0;
  // estimate the object location by expected values
  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  printf("XE: %lf\n", xe); 
  printf("YE: %lf\n", ye);
  double distance = sqrt(pow((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                         pow((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
  FILE *fid;
  fid = fopen("nat_result.txt", "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    abort();
  }
  fprintf(fid, "XE: %lf\n", xe);
  fprintf(fid, "YE: %lf\n", ye);
  fprintf(fid, "distance: %lf\n", distance);
  fclose(fid);
#endif
  // CUDA freeing of memory
  hipFree(weights_GPU);
  hipFree(arrayY_GPU);
  hipFree(arrayX_GPU);

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

  char usage[] = "double.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
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
    seed[i] = (rand() % 10000000) * i;
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

  // malloc matrix
  unsigned char *I =
      (unsigned char *)malloc(sizeof(unsigned char) * IszX * IszY * Nfr);
  // call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  e_compute = std::chrono::high_resolution_clock::now();
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
  free(seed);
  free(I);
  return 0;
}
