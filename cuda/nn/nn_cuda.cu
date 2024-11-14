#include "cuda.h"
#include <chrono>
#include <float.h>
#include <iostream>
#include <stdio.h>
#include <vector>

#define min(a, b) a > b ? b : a
#define ceilDiv(a, b) (a + b - 1) / b
#define print(x) printf(#x ": %lu\n", (unsigned long)x)
//#define DEBUG

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS                                                           \
  28               // character position of the latitude value in each record
#define OPEN 10000 // initial value of nearest neighbors
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;
#define BREAKDOWNS

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
#define WARMUP
typedef struct latLong {
  float lat;
  float lng;
} LatLong;

typedef struct record {
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename, std::vector<Record> &records,
             std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records, float *distances, int numRecords,
                int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char *filename, int *r, float *lat,
                     float *lng, int *q, int *t, int *p, int *d);

/**
 * Kernel
 * Executed on GPU
 * Calculates the Euclidean distance from each record in the database to the
 * target position
 */
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,
                       float lat, float lng) {
  // int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x *
  // blockIdx.x + threadIdx.x;
  int globalId = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) +
                 threadIdx.x; // more efficient
  LatLong *latLong = d_locations + globalId;
  if (globalId < numRecords) {
    float *dist = d_distances + globalId;
    *dist = (float)sqrt((lat - latLong->lat) * (lat - latLong->lat) +
                        (lng - latLong->lng) * (lng - latLong->lng));
  }
}

/**
 * This program finds the k-nearest neighbors
 **/

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();

  int i = 0;
  float lat, lng;
  int quiet = 0, timing = 0, platform = 0, device = 0;

  std::vector<Record> records;
  std::vector<LatLong> locations;
  char filename[100];
  int resultsCount = 10;

  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng, &quiet,
                       &timing, &platform, &device)) {
    printUsage();
    return 0;
  }

  int numRecords = loadData(filename, records, locations);
  if (resultsCount > numRecords)
    resultsCount = numRecords;

  auto end_0 = std::chrono::high_resolution_clock::now();
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  char *warm;
  cudaMalloc((void **)&warm, sizeof(char));
  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  s_compute = std::chrono::high_resolution_clock::now();
  // Pointers to host memory
  float *distances;
  // Pointers to device memory
  LatLong *d_locations;
  float *d_distances;

  // Scaling calculations - added by Sam Kauffman
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  unsigned long maxGridX = deviceProp.maxGridSize[0];
  unsigned long threadsPerBlock =
      min(deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK);
  size_t totalDeviceMemory;
  size_t freeDeviceMemory;
  cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);
  unsigned long usableDeviceMemory =
      freeDeviceMemory * 85 /
      100; // 85% arbitrary throttle to compensate for known CUDA bug
  unsigned long maxThreads =
      usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
  if (numRecords > maxThreads) {
    fprintf(stderr, "Error: Input too large.\n");
    exit(1);
  }
  unsigned long blocks =
      ceilDiv(numRecords, threadsPerBlock); // extra threads will do nothing
  unsigned long gridY = ceilDiv(blocks, maxGridX);
  unsigned long gridX = ceilDiv(blocks, gridY);
  // There will be no more than (gridY - 1) extra blocks
  dim3 gridDim(gridX, gridY);

#ifdef DEBUG
  print(totalDeviceMemory); // 804454400
  print(freeDeviceMemory);
  print(usableDeviceMemory);
  print(maxGridX);                      // 65535
  print(deviceProp.maxThreadsPerBlock); // 1024
  print(threadsPerBlock);
  print(maxThreads);
  print(blocks); // 130933
  print(gridY);
  print(gridX);
#endif

  /**
   * Allocate memory on host and device
   */
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif
  distances = (float *)malloc(sizeof(float) * numRecords);
  cudaMalloc((void **)&d_locations, sizeof(LatLong) * numRecords);
  cudaMalloc((void **)&d_distances, sizeof(float) * numRecords);
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif
  /**
   * Transfer data from host to device
   */
  cudaMemcpy(d_locations, &locations[0], sizeof(LatLong) * numRecords,
             cudaMemcpyHostToDevice);
#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif
  /**
   * Execute kernel
   */
  euclid<<<gridDim, threadsPerBlock>>>(d_locations, d_distances, numRecords,
                                       lat, lng);
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif
  // Copy data from device memory to host memory
  cudaMemcpy(distances, d_distances, sizeof(float) * numRecords,
             cudaMemcpyDeviceToHost);
#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif
  // find the resultsCount least distances
  findLowest(records, distances, numRecords, resultsCount);

  // print out results
#ifdef OUTPUT
  std::cerr << " Store results to output!!" << std::endl;
  // Store the result into a file.
  FILE *fpo = fopen("nat_result.txt", "w");
  for (int i = 0; i < resultsCount; i++)
    fprintf(fpo, "%s --> Distance=%f\n", records[i].recString,
            records[i].distance);
  fclose(fpo);
#endif

  // Free memory
  cudaFree(d_locations);
  cudaFree(d_distances);
  e_compute = std::chrono::high_resolution_clock::now();

  free(distances);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli_0 = end_0 - start_0;
  std::cerr << "Init time: " << elapsed_milli_0.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> elapsed_milli = end - start;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
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
#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
  // free warmup
  cudaFree(warm);
#endif
}

int loadData(char *filename, std::vector<Record> &records,
             std::vector<LatLong> &locations) {
  FILE *flist, *fp;
  int i = 0;
  char dbname[64];
  int recNum = 0;

  /**Main processing **/

  flist = fopen(filename, "r");
  while (!feof(flist)) {
    /**
     * Read in all records of length REC_LENGTH
     * If this is the last file in the filelist, then done
     * else open next file to be read next iteration
     */
    if (fscanf(flist, "%s\n", dbname) != 1) {
      fprintf(stderr, "error reading filelist\n");
      exit(0);
    }
    fp = fopen(dbname, "r");
    if (!fp) {
      printf("error opening a db\n");
      exit(1);
    }
    // read each record
    while (!feof(fp)) {
      Record record;
      LatLong latLong;
      fgets(record.recString, 49, fp);
      fgetc(fp); // newline
      if (feof(fp))
        break;

      // parse for lat and long
      char substr[6];

      for (i = 0; i < 5; i++)
        substr[i] = *(record.recString + i + 28);
      substr[5] = '\0';
      latLong.lat = atof(substr);

      for (i = 0; i < 5; i++)
        substr[i] = *(record.recString + i + 33);
      substr[5] = '\0';
      latLong.lng = atof(substr);

      locations.push_back(latLong);
      records.push_back(record);
      recNum++;
    }
    fclose(fp);
  }
  fclose(flist);
  //    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
  return recNum;
}

void findLowest(std::vector<Record> &records, float *distances, int numRecords,
                int topN) {
  int i, j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for (i = 0; i < topN; i++) {
    minLoc = i;
    for (j = i; j < numRecords; j++) {
      val = distances[j];
      if (val < distances[minLoc])
        minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char *filename, int *r, float *lat,
                     float *lng, int *q, int *t, int *p, int *d) {
  int i;
  if (argc < 2)
    return 1; // error
  strncpy(filename, argv[1], 100);
  char flag;

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
      case 'r': // number of results
        i++;
        *r = atoi(argv[i]);
        break;
      case 'l':                  // lat or lng
        if (argv[i][2] == 'a') { // lat
          *lat = atof(argv[i + 1]);
        } else { // lng
          *lng = atof(argv[i + 1]);
        }
        i++;
        break;
      case 'h': // help
        return 1;
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
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] "
         "[-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
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
