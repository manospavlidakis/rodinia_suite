#include "work_group_size.h"

#define LIMIT -999

#include "../common/opencl_util.h"
#include <CL/cl.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifdef NO_INTERLEAVE
#include "CL/cl_ext.h"
#endif
std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
// global variables

int blosum62[24][24] = {{4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
                         -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4},
                        {-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
                         -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4},
                        {-2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
                         -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4},
                        {-2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
                         -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4},
                        {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
                         -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
                         0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4},
                        {-1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
                         -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
                         -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
                         -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
                         1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
                         2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},
                        {-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
                         -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4},
                        {-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
                         5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
                         0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
                         -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
                         -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4},
                        {0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
                         -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
                         -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
                         -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
                         1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4},
                        {-2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
                         -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4},
                        {-1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
                         -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

// local variables
static cl_context context;
static cl_command_queue cmd_queue;
static cl_command_queue cmd_queue2;
static cl_device_type device_type;
static cl_device_id *device_list;
static cl_int num_devices;

static int initialize() {
  size_t size;
  cl_int result, error;
  cl_uint platformCount;
  cl_platform_id *platforms = NULL;
  cl_context_properties ctxprop[3];

  display_device_info(&platforms, &platformCount);
  select_device_type(platforms, &platformCount, &device_type);
  validate_selection(platforms, &platformCount, ctxprop, &device_type);

  // create OpenCL context
  context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, &error);
  if (!context) {
    printf("ERROR: clCreateContextFromType(%s) failed with error code %d.\n",
           (device_type == CL_DEVICE_TYPE_ACCELERATOR) ? "FPGA"
           : (device_type == CL_DEVICE_TYPE_GPU)       ? "GPU"
                                                       : "CPU",
           error);
    display_error_message(error, stdout);
    return -1;
  }

  // get the list of GPUs
  result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
  num_devices = (int)(size / sizeof(cl_device_id));

  if (result != CL_SUCCESS || num_devices < 1) {
    printf("ERROR: clGetContextInfo() failed\n");
    return -1;
  }
  device_list = new cl_device_id[num_devices];
  if (!device_list) {
    printf("ERROR: new cl_device_id[] failed\n");
    return -1;
  }
  result =
      clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
  if (result != CL_SUCCESS) {
    printf("ERROR: clGetContextInfo() failed\n");
    return -1;
  }

  // create command queue for the first device
  cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);
  if (!cmd_queue) {
    printf("ERROR: clCreateCommandQueue() failed\n");
    return -1;
  }

  // create command queue for the first device
  cmd_queue2 = clCreateCommandQueue(context, device_list[0], 0, NULL);
  if (!cmd_queue) {
    printf("ERROR: clCreateCommandQueue() failed\n");
    return -1;
  }

  free(platforms); // platforms isn't needed in the main function

  return 0;
}

static int shutdown() {
  // release resources
  if (cmd_queue)
    clReleaseCommandQueue(cmd_queue);
  if (context)
    clReleaseContext(context);
  if (device_list)
    delete device_list;

  // reset all variables
  cmd_queue = 0;
  context = 0;
  device_list = 0;
  num_devices = 0;
  device_type = 0;

  return 0;
}

int maximum(int a, int b, int c) {
  int k;
  if (a <= b)
    k = b;
  else
    k = a;
  if (k <= c)
    return (c);
  else
    return (k);
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <kernel_version> \n",
          argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
  fprintf(stderr,
          "\t<kernel_version> - version of kernel or bianry file to load\n");
  exit(1);
}

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  int max_rows, max_cols, penalty;
  char *version_string;
  int version_number;
  size_t sourcesize;

  version_number = 0;
  version_string = "v0";

  init_fpga2(&argc, &argv, &version_string, &version_number);

  int enable_traceback = 0;

  if (argc >= 3) {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
  } else {
    usage(argc, argv);
    exit(1);
  }

  // the lengths of the two sequences should be divisible by 16.
  // And at current stage, max_rows needs to equal max_cols
  if (is_ndrange_kernel(version_number)) {
    if (atoi(argv[1]) % 16 != 0) {
      fprintf(stderr, "The dimension values must be a multiple of 16\n");
      exit(1);
    }
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  int num_rows = max_rows;
  int num_cols = max_cols;

  int data_size = max_cols * max_rows;
  int ref_size = num_cols * num_rows;
  int *reference = (int *)alignedMalloc(ref_size * sizeof(int));
  int *input_itemsets = (int *)alignedMalloc(data_size * sizeof(int));
  int *output_itemsets = (int *)alignedMalloc(ref_size * sizeof(int));
  // for v7 and above
  int *buffer_v = NULL, *buffer_h = NULL;
  srand(7);

  // initialization
  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  for (int i = 1; i < max_rows; i++) { // initialize the first column
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  }

  for (int j = 1; j < max_cols; j++) { // initialize the first row
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      reference[i * max_cols + j] =
          blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (int i = 1; i < max_rows; i++) {
    input_itemsets[i * max_cols] = -i * penalty;
  }
  for (int j = 1; j < max_cols; j++) {
    input_itemsets[j] = -j * penalty;
  }

  // read the kernel core source
  s_init_fpga_timer = std::chrono::high_resolution_clock::now();
  // get name of kernel file based on version
  char *kernel_file_path =
      getVersionedKernelName2("./nw_kernel", version_string);
  char *source = read_kernel(kernel_file_path, &sourcesize);

  char const *kernel_nw1 = "nw_kernel1";
  char const *kernel_nw2 = "nw_kernel2";

  int nworkitems, workgroupsize = 0;
  nworkitems = BSIZE;

  if (nworkitems < 1 || workgroupsize < 0) {
    printf("ERROR: invalid or missing <num_work_items>[/<work_group_size>]\n");
    return -1;
  }
  // set global and local workitems
  size_t local_work[3] = {(size_t)((workgroupsize > 0) ? workgroupsize : 1), 1,
                          1};
  size_t global_work[3] = {(size_t)nworkitems, 1, 1};

  // OpenCL initialization
  if (initialize()) {
    return -1;
  }
  cl_int err = 0;
  // compile kernel
  cl_program prog =
      clCreateProgramWithBinary(context, 1, device_list, &sourcesize,
                                (const unsigned char **)&source, NULL, &err);
#ifdef DEBUG
  if (err != CL_SUCCESS) {
    printf("ERROR: clCreateProgramWithSource/Binary() => %d\n", err);
    display_error_message(err, stderr);
    return -1;
  }
#endif
  char clOptions[110];
  sprintf(clOptions, "-I .");

#ifdef USE_JIT
  sprintf(clOptions + strlen(clOptions), " -DBSIZE=%d -DPAR=%d", BSIZE, PAR);
#endif

  clBuildProgram_SAFE(prog, num_devices, device_list, clOptions, NULL, NULL);

  cl_kernel kernel1;
  cl_kernel kernel2;
  kernel1 = clCreateKernel(prog, kernel_nw1, &err);
  if (is_ndrange_kernel(version_number)) {
    kernel2 = clCreateKernel(prog, kernel_nw2, &err);
  } else {
    // use the same kernel in single work-item versions
    kernel2 = kernel1;
  }

  if (err != CL_SUCCESS) {
    printf("ERROR: clCreateKernel() 0 => %d\n", err);
    return -1;
  }
  clReleaseProgram(prog);

  e_init_fpga_timer = std::chrono::high_resolution_clock::now();
  auto end_0 = std::chrono::high_resolution_clock::now();
  s_compute = std::chrono::high_resolution_clock::now();
  // create buffers
  cl_mem reference_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      ref_size * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("ERROR: clCreateBuffer reference (size:%d) => %d\n", ref_size, err);
    return -1;
  }
  int device_buff_size = (version_number >= 7)
                             ? num_cols * (num_rows + 1)
                             : ((version_number == 5) ? ref_size : data_size);
  cl_mem input_itemsets_d = clCreateBuffer(
      context, CL_MEM_READ_WRITE, device_buff_size * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n",
           device_buff_size, err);
    return -1;
  }

  // write buffers
  CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue, reference_d, 1, 0,
                                    ref_size * sizeof(int), reference, 0, 0,
                                    0));
  CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue, input_itemsets_d, 1, 0,
                                    data_size * sizeof(int), input_itemsets, 0,
                                    0, 0));

  int worksize = max_cols - 1;
  // these two parameters are for extension use, don't worry about it.
  int offset_r = 0, offset_c = 0;
  int block_width = worksize / BSIZE;

  // constant kernel arguments
  CL_SAFE_CALL(
      clSetKernelArg(kernel1, 0, sizeof(void *), (void *)&reference_d));
  CL_SAFE_CALL(
      clSetKernelArg(kernel1, 1, sizeof(void *), (void *)&input_itemsets_d));
  CL_SAFE_CALL(clSetKernelArg(kernel1, 2, sizeof(cl_int), (void *)&num_rows));
  CL_SAFE_CALL(clSetKernelArg(kernel1, 3, sizeof(cl_int), (void *)&penalty));
  CL_SAFE_CALL(clSetKernelArg(kernel1, 5, sizeof(cl_int), (void *)&offset_r));
  CL_SAFE_CALL(clSetKernelArg(kernel1, 6, sizeof(cl_int), (void *)&offset_c));

  CL_SAFE_CALL(
      clSetKernelArg(kernel2, 0, sizeof(void *), (void *)&reference_d));
  CL_SAFE_CALL(
      clSetKernelArg(kernel2, 1, sizeof(void *), (void *)&input_itemsets_d));
  CL_SAFE_CALL(clSetKernelArg(kernel2, 2, sizeof(cl_int), (void *)&num_cols));
  CL_SAFE_CALL(clSetKernelArg(kernel2, 3, sizeof(cl_int), (void *)&penalty));
  CL_SAFE_CALL(
      clSetKernelArg(kernel2, 5, sizeof(cl_int), (void *)&block_width));
  CL_SAFE_CALL(clSetKernelArg(kernel2, 6, sizeof(cl_int), (void *)&offset_r));
  CL_SAFE_CALL(clSetKernelArg(kernel2, 7, sizeof(cl_int), (void *)&offset_c));

  // NDRange versions
  for (int blk = 1; blk <= worksize / BSIZE; blk++) {
    global_work[0] = BSIZE * blk;
    local_work[0] = BSIZE;

    CL_SAFE_CALL(clSetKernelArg(kernel1, 4, sizeof(cl_int), (void *)&blk));
    CL_SAFE_CALL(clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, NULL,
                                        global_work, local_work, 0, 0, NULL));
  }

  for (int blk = worksize / BSIZE - 1; blk >= 1; blk--) {
    global_work[0] = BSIZE * blk;
    local_work[0] = BSIZE;

    CL_SAFE_CALL(clSetKernelArg(kernel2, 4, sizeof(cl_int), (void *)&blk));
    CL_SAFE_CALL(clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, NULL,
                                        global_work, local_work, 0, 0, NULL));
  }

  err = clEnqueueReadBuffer(cmd_queue, input_itemsets_d, 1, 0,
                            ref_size * sizeof(int), output_itemsets, 0, 0, 0);
  e_compute = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
  FILE *fpo = fopen("result.txt", "w");
  fprintf(fpo, "max_cols: %d, penalty: %d\n", max_cols - 1, penalty);
  for (int i = max_cols - 2, j = max_rows - 2; i >= 0 && j >= 0;) {
    fprintf(fpo, "[%d, %d] ", i, j);
    int nw = 0, n = 0, w = 0, traceback;
    if (i == 0 && j == 0) {
      fprintf(fpo, "(output: %d)\n", output_itemsets[0]);
      break;
    }
    if (i > 0 && j > 0) {
      nw = output_itemsets[(i - 1) * max_cols + j - 1];
      w = output_itemsets[i * max_cols + j - 1];
      n = output_itemsets[(i - 1) * max_cols + j];
      fprintf(fpo, "(nw: %d, w: %d, n: %d, ref: %d) ", nw, w, n,
              reference[i * max_cols + j]);
    } else if (i == 0) {
      nw = n = LIMIT;
      w = output_itemsets[i * max_cols + j - 1];
    } else if (j == 0) {
      nw = w = LIMIT;
      n = output_itemsets[(i - 1) * max_cols + j];
    } else {
    }

    // traceback = maximum(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + reference[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if (traceback != output_itemsets[i * max_cols + j]) {
      fprintf(stderr,
              "Mismatch at (%d, %d). traceback: %d, output_itemsets: %d\n", i,
              j, traceback, output_itemsets[i * max_cols + j]);
      // exit(1);
    }
    fprintf(fpo, "(output: %d)", traceback);

    if (traceback == new_nw) {
      traceback = nw;
      fprintf(fpo, "(->nw) ");
    } else if (traceback == new_w) {
      traceback = w;
      fprintf(fpo, "(->w) ");
    } else if (traceback == new_n) {
      traceback = n;
      fprintf(fpo, "(->n) ");
    } else {
      fprintf(stderr, "Error: inconsistent traceback at (%d, %d)\n", i, j);
      abort();
    }

    fprintf(fpo, "\n");

    if (traceback == nw) {
      i--;
      j--;
      continue;
    }

    else if (traceback == w) {
      j--;
      continue;
    }

    else if (traceback == n) {
      i--;
      continue;
    }

    else
      ;
  }

  fclose(fpo);
  printf("Traceback saved in result.txt\n");
#endif

  clReleaseMemObject(reference_d);
  clReleaseMemObject(input_itemsets_d);

  free(reference);
  free(input_itemsets);
  free(output_itemsets);
  free(source);

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
  // OpenCL shutdown
  if (shutdown())
    return -1;
}
