#include "../common/opencl_util.h"
#include "CL_helper.h"
#include "hotspot3D_common.h"
#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL (0.001)
#define STR_SIZE (256)
#define MAX_PD (3.0e6)

/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100

/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;

float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
float amb_temp = 80.0;

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <rows> <cols> <layers> <iterations> <powerFile> "
          "<tempFile> <outputFile>\n",
          argv[0]);
  fprintf(stderr,
          "\t<rows>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr,
          "\t<cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr,
          "\t<layers>     - number of layers in the grid (positive integer)\n");
  fprintf(stderr, "\t<iteration>  - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial "
                  "power values of each cell\n");
  fprintf(stderr, "\t<tempFile>   - name of the file containing the initial "
                  "temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile> - output file\n\n");

  fprintf(stderr, "\tNote: If input file names are not supplied, input is "
                  "generated randomly.\n");
  fprintf(stderr, "\tNote: If output file name is not supplied, output will "
                  "not be written to disk.\n");
  exit(1);
}

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();

  int write_out = 0;
  int version = 0;

  init_fpga(&argc, &argv, &version);

  if (argc < 5 || argc > 9) {
    usage(argc, argv);
  }

  char *pfile = NULL, *tfile = NULL, *ofile = NULL;
  int iterations = atoi(argv[3]);

  pfile = argv[4];
  tfile = argv[5];
  ofile = argv[6];

  int numCols = atoi(argv[1]);
  int numRows = atoi(argv[1]);
  int layers = atoi(argv[2]);

  /* calculating parameters*/

  float dx = chip_height / numRows;
  float dy = chip_width / numCols;
  float dz = t_chip / layers;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx = dy / (2.0 * K_SI * t_chip * dx);
  float Ry = dx / (2.0 * K_SI * t_chip * dy);
  float Rz = dz / (K_SI * dx * dy);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;

  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

  cl_int err;
  size_t devices_size;
  int size = (version >= 7) ? numCols * numRows * layers + PAD
                            : numCols * numRows * layers;
  float *tIn = (float *)alignedCalloc(size, sizeof(float));
  float *pIn = (float *)alignedCalloc(size, sizeof(float));
  float *tempCopy = (float *)alignedMalloc(size * sizeof(float));
  float *tempOut = (float *)alignedCalloc(size, sizeof(float));
  int count = size;
  readinput(tIn, numRows, numCols, layers, tfile);
  readinput(pIn, numRows, numCols, layers, pfile);

  size_t global[2];
  size_t local[2];
  memcpy(tempCopy, tIn, size * sizeof(float));
  s_init_fpga_timer = std::chrono::high_resolution_clock::now();
  cl_context context;
  cl_command_queue command_queue;
  cl_command_queue command_queue2 = NULL;
  cl_program program;
  cl_kernel hotspot3D = NULL;
  cl_kernel ReadKernel = NULL;
  cl_kernel WriteKernel = NULL;
  cl_device_type device_type;

  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;

  // const char *KernelSource = load_kernel_source("hotspotKernel.cl");
  size_t sourcesize;
  char *kernel_file_path =
      getVersionedKernelName("./hotspot3D_kernel", version);
  char *KernelSource = read_kernel(kernel_file_path, &sourcesize);
  free(kernel_file_path);

  cl_platform_id *platforms = NULL;
  cl_uint num_platforms = 0;
  cl_context_properties ctxprop[3];

  display_device_info(&platforms, &num_platforms);
  select_device_type(platforms, &num_platforms, &device_type);
  validate_selection(platforms, &num_platforms, ctxprop, &device_type);

  context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create context from type!\n%s\n", err_code(err));
    return EXIT_FAILURE;
  }

  err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
  int num_devices = (int)(devices_size / sizeof(cl_device_id));
  if (err != CL_SUCCESS || num_devices < 1) {
    printf("Error: Failed to get context info!\n");
    return -1;
  }

  cl_device_id *device_list =
      (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  if (!device_list) {
    printf("Error: Failed to create device list!\n");
    return -1;
  }

  CL_SAFE_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size,
                                device_list, NULL));

  command_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
  if (!command_queue) {
    printf("Error: Failed to create command_queue!\n%s\n", err_code(err));
    return EXIT_FAILURE;
  }
  program = clCreateProgramWithBinary(context, 1, device_list, &sourcesize,
                                      (const unsigned char **)&KernelSource,
                                      NULL, &err);
  if (!program) {
    printf("Error: Failed to create compute program!\n%s\n", err_code(err));
    return EXIT_FAILURE;
  }

  clBuildProgram_SAFE(program, num_devices, device_list, NULL, NULL, NULL);

  hotspot3D = clCreateKernel(program, "hotspotOpt1", &err);
  if (!hotspot3D || err != CL_SUCCESS) {
    printf("Error: Failed to create hotspot3D kernel!\n%s\n", err_code(err));
    return EXIT_FAILURE;
  }
  e_init_fpga_timer = std::chrono::high_resolution_clock::now();
  auto end_0 = std::chrono::high_resolution_clock::now();
  s_compute = std::chrono::high_resolution_clock::now();
  d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL,
                       NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL,
                       NULL);
  d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL,
                       NULL);

  if (!d_a || !d_b || !d_c) {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }

  err = clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0,
                             sizeof(float) * count, tIn, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write tIn to source array!\n%s\n", err_code(err));
    exit(1);
  }

  err = clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0,
                             sizeof(float) * count, pIn, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write pIn to source array!\n%s\n", err_code(err));
    exit(1);
  }

  err = clEnqueueWriteBuffer(command_queue, d_c, CL_TRUE, 0,
                             sizeof(float) * count, tempOut, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write tempOut to source array!\n%s\n",
           err_code(err));
    exit(1);
  }

  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 0, sizeof(cl_mem), &d_b));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 3, sizeof(float), &stepDivCap));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 4, sizeof(int), &numCols));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 5, sizeof(int), &numRows));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 6, sizeof(int), &layers));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 7, sizeof(float), &ce));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 8, sizeof(float), &cw));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 9, sizeof(float), &cn));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 10, sizeof(float), &cs));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 11, sizeof(float), &ct));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 12, sizeof(float), &cb));
  CL_SAFE_CALL(clSetKernelArg(hotspot3D, 13, sizeof(float), &cc));

  int j;
  for (j = 0; j < iterations; j++) {
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 1, sizeof(cl_mem), &d_a));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 2, sizeof(cl_mem), &d_c));

    global[0] = numCols;
    global[1] = numRows;

    local[0] = WG_SIZE_X;
    local[1] = WG_SIZE_Y;

    CL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, hotspot3D, 2, NULL,
                                        global, local, 0, NULL, NULL));
    cl_mem temp = d_a;
    d_a = d_c;
    d_c = temp;
  }

  // pointers are always swapped one extra time at the end of the iteration loop
  // and hence, d_a points to the output, not d_c
  err = clEnqueueReadBuffer(command_queue, d_a, CL_TRUE, 0,
                            sizeof(float) * count, tempOut, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }

#ifdef DEBUG
  float *answer = (float *)calloc(size, sizeof(float));
  computeTempCPU(pIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry,
                 Rz, dt, amb_temp, iterations);

  // for an even number of iterations, "tempCopy" will point to correct output
  // of the CPU function and for an odd number, "answer" will
  float *CPUOut = (iterations % 2 == 1) ? answer : tempCopy;
  float acc =
      (version >= 7)
          ? accuracy(tempOut + PAD, CPUOut + PAD, numRows * numCols * layers)
          : accuracy(tempOut, CPUOut, numRows * numCols * layers);

  printf("Accuracy: %e\n", acc);
  if (write_out)
    writeoutput(tempOut, numRows, numCols, layers, ofile);

#endif

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  e_compute = std::chrono::high_resolution_clock::now();
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
  clReleaseProgram(program);
  clReleaseKernel(hotspot3D);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
