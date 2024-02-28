#ifndef __NEAREST_NEIGHBOR__
#define __NEAREST_NEIGHBOR__

#include "nearestNeighbor.h"
#include "../common/opencl_util.h"
#include <chrono>
#include <iostream>

cl_context context = NULL;
cl_kernel NN_kernel;
cl_int status;
cl_program cl_NN_program;
cl_command_queue command_queue;

std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
int version;

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  std::vector<Record> records;
  float *recordDistances;
  // LatLong locations[REC_WINDOW];
  std::vector<LatLong> locations;
  int i;

  // args
  char filename[100];
  int resultsCount = 10, quiet = 0, timing = 1, platform = -1, device = -1;
  float lat = 0.0, lng = 0.0;

  init_fpga(&argc, &argv, &version);

  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng, &quiet,
                       &timing, &platform, &device)) {
    printUsage();
    return 0;
  }

  int numRecords = loadData(filename, records, locations);

  // for(i=0;i<numRecords;i++)
  //     printf("%s, %f,
  //     %f\n",(records[i].recString),locations[i].lat,locations[i].lng);
#ifdef DEBUG
  printf("Number of records: %d\n", numRecords);
  printf("Finding the %d closest neighbors.\n", resultsCount);
#endif

  if (resultsCount > numRecords)
    resultsCount = numRecords;
  s_init_fpga_timer = std::chrono::high_resolution_clock::now();
  context = cl_init_context(platform, device, quiet);
  // 1. set up kernel
  // get name of kernel file based on version
  char *kernel_file_path = getVersionedKernelName("./nn_kernel", version);

#ifdef USE_JIT
  cl_NN_program = cl_compileProgram(kernel_file_path, (char *)"-I .");
#else
  cl_NN_program = cl_compileProgram(kernel_file_path, NULL);
#endif

  NN_kernel = clCreateKernel(cl_NN_program, "NearestNeighbor", &status);
#ifdef DEBUG
  status =
      cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel", true);
  if (status)
    exit(1);
#endif
  e_init_fpga_timer = std::chrono::high_resolution_clock::now();
  auto end_0 = std::chrono::high_resolution_clock::now();
  s_compute = std::chrono::high_resolution_clock::now();
  recordDistances = OpenClFindNearestNeighbors(context, numRecords, locations,
                                               lat, lng, timing);

  // find the resultsCount least distances
  findLowest(records, recordDistances, numRecords, resultsCount);
  e_compute = std::chrono::high_resolution_clock::now();

  // print out results
#ifdef DEBUG
  for (i = 0; i < resultsCount; i++) {
    printf("%s --> Distance=%f\n", records[i].recString, records[i].distance);
#endif

    free(recordDistances);
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
              << elapsed_milli.count() - prep_milli.count() << " ms"
              << std::endl;

    clReleaseProgram(cl_NN_program);
    clReleaseKernel(NN_kernel);
    clReleaseCommandQueue(command_queue);

    clReleaseContext(context);
    return 0;
  }

  float *OpenClFindNearestNeighbors(cl_context context, int numRecords,
                                    std::vector<LatLong> &locations, float lat,
                                    float lng, int timing) {

    // 2. set up memory on device and send ipts data to device
    // copy ipts(1,2) to device
    // also need to alloate memory for the distancePoints
    cl_mem d_locations;
    cl_mem d_distances;

    cl_int error = 0;

    d_locations = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(LatLong) * numRecords, NULL, &error);

    d_distances = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * numRecords, NULL, &error);

    command_queue = cl_getCommandQueue();
    cl_event writeEvent, kernelEvent, readEvent;
    error = clEnqueueWriteBuffer(command_queue, d_locations,
                                 1, // change to 0 for nonblocking write
                                 0, // offset
                                 sizeof(LatLong) * numRecords, &locations[0], 0,
                                 NULL, &writeEvent);

    // 3. send arguments to device
    cl_int argchk;
    argchk = clSetKernelArg(NN_kernel, 0, sizeof(cl_mem), (void *)&d_locations);
    argchk |=
        clSetKernelArg(NN_kernel, 1, sizeof(cl_mem), (void *)&d_distances);
    argchk |= clSetKernelArg(NN_kernel, 2, sizeof(int), (void *)&numRecords);
    argchk |= clSetKernelArg(NN_kernel, 3, sizeof(float), (void *)&lat);
    argchk |= clSetKernelArg(NN_kernel, 4, sizeof(float), (void *)&lng);
#ifdef DEBUG
    cl_errChk(argchk, "ERROR in Setting Nearest Neighbor kernel args", true);
#endif
    // 4. enqueue kernel
    size_t globalWorkSize[1];
    globalWorkSize[0] = numRecords;
    if (numRecords % 64)
      globalWorkSize[0] += 64 - (numRecords % 64);
    // printf("Global Work Size: %zu\n",globalWorkSize[0]);

    if (is_ndrange_kernel(version)) {
      error =
          clEnqueueNDRangeKernel(command_queue, NN_kernel, 1, 0, globalWorkSize,
                                 NULL, 0, NULL, &kernelEvent);
    } else {
      error = clEnqueueTask(command_queue, NN_kernel, 0, NULL, &kernelEvent);
    }
#ifdef DEBUG
    cl_errChk(error, "ERROR in Executing Kernel NearestNeighbor", true);
#endif
    // 5. transfer data off of device
    // create distances std::vector
    float *distances = (float *)alignedMalloc(sizeof(float) * numRecords);

    error = clEnqueueReadBuffer(command_queue, d_distances,
                                1, // change to 0 for nonblocking write
                                0, // offset
                                sizeof(float) * numRecords, distances, 0, NULL,
                                &readEvent);

#ifdef DEBUG
    cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
#endif
    // 6. return finalized data and release buffers
    clReleaseMemObject(d_locations);
    clReleaseMemObject(d_distances);
    return distances;
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
       * Read in REC_WINDOW records of length REC_LENGTH
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
    return recNum;
  }

  void findLowest(std::vector<Record> & records, float *distances,
                  int numRecords, int topN) {
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

  int parseCommandline(int argc, char *argv[], char *filename, int *r,
                       float *lat, float *lng, int *q, int *t, int *p, int *d) {
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
          break;
        case 'q': // quiet
          *q = 1;
          break;
        case 't': // timing
          *t = 1;
          break;
        case 'p': // platform
          i++;
          *p = 1;
          // atoi(argv[i]);
          break;
        case 'd': // device
          i++;
          *d = 0;
          // atoi(argv[i]);
          break;

        default:
          return 1; // error
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
    printf(
        "nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] "
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
    printf("-v[int]      Choose kernel version, must be at the end of argument "
           "list)\n");
    printf("Notes: 1. The filename is required as the first parameter.\n");
    printf("       2. If you declare either the device or the platform,\n");
    printf("          you must declare both.\n\n");
  }

#endif
