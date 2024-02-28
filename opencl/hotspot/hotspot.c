#include "hotspot.h"
#include <chrono>
#include <iostream>
cl_device_id *device_list;
std::chrono::high_resolution_clock::time_point s_init_fpga_timer;
std::chrono::high_resolution_clock::time_point e_init_fpga_timer;

std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {
  int i, j, index = 0;
  FILE *fp;
  char str[STR_SIZE];

  if ((fp = fopen(file, "w")) == 0)
    printf("The file was not opened\n");

  for (i = 0; i < grid_rows; i++)
    for (j = 0; j < grid_cols; j++) {

      sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
      fputs(str, fp);
      index++;
    }
  fclose(fp);
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i, j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  // printf("Reading %s\n", file);

  if ((fp = fopen(file, "r")) == 0)
    fatal("The input file was not opened");

  for (i = 0; i <= grid_rows - 1; i++)
    for (j = 0; j <= grid_cols - 1; j++) {
      if (fgets(str, STR_SIZE, fp) == NULL)
        fatal("Error reading file\n");
      if (feof(fp))
        fatal("not enough lines in file");
      // if ((sscanf(str, "%d%f", &index, &val) != 2) || (index !=
      // ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1))
        fatal("invalid file format");
      vect[i * grid_cols + j] = val;
    }

  fclose(fp);
}

/*
  compute N time steps
*/

int compute_tran_temp(cl_mem MatrixPower, cl_mem MatrixTemp[2], int col,
                      int row, int total_iterations, int pyramid_height,
                      int blockCols, int blockRows, int haloCols, int haloRows,
                      int version_number, int block_size_x, int block_size_y) {
  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;

  int src = 0, dst = 1;

  // Determine GPU work group grid
  size_t global_work_size[2];
  global_work_size[0] = block_size_x * blockCols;
  // global_work_size[0] = 1;
  global_work_size[1] = block_size_y * blockRows;
  // global_work_size[1] = 1;
  size_t local_work_size[2];
  // local_work_size[0] = block_size_x;
  local_work_size[0] = 8;
  // local_work_size[1] = block_size_y;
  local_work_size[1] = 8;

  CL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&MatrixPower));
  CL_SAFE_CALL(clSetKernelArg(kernel, 4, sizeof(int), (void *)&col));
  CL_SAFE_CALL(clSetKernelArg(kernel, 5, sizeof(int), (void *)&row));
  CL_SAFE_CALL(clSetKernelArg(kernel, 6, sizeof(int), (void *)&haloCols));
  CL_SAFE_CALL(clSetKernelArg(kernel, 7, sizeof(int), (void *)&haloRows));
  CL_SAFE_CALL(clSetKernelArg(kernel, 8, sizeof(float), (void *)&Cap));
  CL_SAFE_CALL(clSetKernelArg(kernel, 9, sizeof(float), (void *)&Rx));
  CL_SAFE_CALL(clSetKernelArg(kernel, 10, sizeof(float), (void *)&Ry));
  CL_SAFE_CALL(clSetKernelArg(kernel, 11, sizeof(float), (void *)&Rz));
  CL_SAFE_CALL(clSetKernelArg(kernel, 12, sizeof(float), (void *)&step));

  // Launch kernel
  int t;
  for (t = 0; t < total_iterations; t += pyramid_height) {
    int iter = MIN(pyramid_height, total_iterations - t);
    CL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(int), (void *)&iter));
    CL_SAFE_CALL(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&MatrixTemp[src]));
    CL_SAFE_CALL(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&MatrixTemp[dst]));
    // Launch kernel
    CL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                        global_work_size, local_work_size, 0,
                                        NULL, NULL));
    src = 1 - src;
    dst = 1 - dst;
  }

  return src;
}

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> "
          "<temp_file> <power_file> <output_file>\n",
          argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid "
                  "(positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid height(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial "
                  "temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated "
                  "power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file (optional)\n");
  fprintf(stderr, "\t<bitstream version> eg. 2, 7\n");
  fprintf(stderr, "\t<bistream version string>e. v2, v7\n");

  fprintf(stderr, "\tNote: If output file name is not supplied, output will "
                  "not be written to disk.\n");
  exit(1);
}

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  int write_out = 0;
  char *version_string;
  int version_number;

  version_number = 0;
  version_string = "v0";

  init_fpga2(&argc, &argv, &version_string, &version_number);

  int size;
  int grid_rows, grid_cols = 0;
  float *FilesavingTemp = NULL, *FilesavingPower = NULL;
  char *tfile, *pfile, *ofile = NULL;
  int block_size_x = BLOCK_X;
  int block_size_y = BLOCK_Y;

  int total_iterations = 60;
  int pyramid_height = 1; // number of combined iterations

  if (argc < 5)
    usage(argc, argv);

  if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[1])) <= 0 ||
      (pyramid_height = atoi(argv[2])) <= 0 ||
      (total_iterations = atoi(argv[3])) < 0)
    usage(argc, argv);

  tfile = argv[4];
  pfile = argv[5];
  ofile = argv[6];

  size = grid_rows * grid_cols;
  FilesavingTemp = (float *)alignedMalloc(size * sizeof(float));
  FilesavingPower = (float *)alignedMalloc(size * sizeof(float));

  if (!FilesavingPower || !FilesavingTemp)
    fatal("unable to allocate memory");

  // Read input data from disk
  readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
  readinput(FilesavingPower, grid_rows, grid_cols, pfile);

  // --------------- pyramid parameters ---------------
  int haloCols = (pyramid_height)*EXPAND_RATE / 2;
  int haloRows = (pyramid_height)*EXPAND_RATE / 2;
  int smallBlockCol = block_size_x - (pyramid_height)*EXPAND_RATE;
  int smallBlockRow = block_size_y - (pyramid_height)*EXPAND_RATE;
  int blockCols =
      grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
  int blockRows =
      grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

  // Copy final temperature data back
  float *MatrixOut = NULL;

  MatrixOut = (float *)alignedMalloc(sizeof(float) * size);

  s_init_fpga_timer = std::chrono::high_resolution_clock::now();
  size_t devices_size;
  cl_int result, error;
  cl_uint platformCount;
  cl_platform_id *platforms = NULL;
  cl_context_properties ctxprop[3];
  cl_device_type device_type;

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
  result =
      clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
  int num_devices = (int)(devices_size / sizeof(cl_device_id));

  if (result != CL_SUCCESS || num_devices < 1) {
    printf("ERROR: clGetContextInfo() failed\n");
    return -1;
  }
  device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  if (!device_list) {
    printf("ERROR: new cl_device_id[] failed\n");
    return -1;
  }
  CL_SAFE_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size,
                                device_list, NULL));
  device = device_list[0];

  // Create command queue
  command_queue =
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  if (version_number >= 7) {
    command_queue2 = clCreateCommandQueue(context, device,
                                          CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS)
      fatal_CL(error, __LINE__);
  }

  // Load kernel source from file
  char *kernel_file_path =
      getVersionedKernelName2("hotspot_kernel", version_string);

  size_t sourceSize;
  char *source = read_kernel(kernel_file_path, &sourceSize);

  cl_program program =
      clCreateProgramWithBinary(context, 1, device_list, &sourceSize,
                                (const unsigned char **)&source, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  char clOptions[110];
  sprintf(clOptions, "-I.");
  // Create an executable from the kernel
  // clBuildProgram_SAFE(program, 1, &device, clOptions, NULL, NULL);
  //
  kernel = clCreateKernel(program, "hotspot", &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  e_init_fpga_timer = std::chrono::high_resolution_clock::now();

  auto end_0 = std::chrono::high_resolution_clock::now();
  s_compute = std::chrono::high_resolution_clock::now();
  // Create two temperature matrices and copy the temperature input data
  cl_mem MatrixTemp[2], MatrixPower = NULL;

  // Create input memory buffers on device
  MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  CL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, MatrixTemp[0], CL_TRUE, 0,
                                    sizeof(float) * size, FilesavingTemp, 0,
                                    NULL, NULL));

  // Copy the power input data
  MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size,
                               NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  CL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, MatrixPower, CL_TRUE, 0,
                                    sizeof(float) * size, FilesavingPower, 0,
                                    NULL, NULL));

  // Perform the computation
  int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows,
                              total_iterations, pyramid_height, blockCols,
                              blockRows, haloCols, haloRows, version_number,
                              block_size_x, block_size_y);
  CL_SAFE_CALL(clEnqueueReadBuffer(command_queue, MatrixTemp[ret], CL_TRUE, 0,
                                   sizeof(float) * size, MatrixOut, 0, NULL,
                                   NULL));

#ifdef DEBUG
  writeoutput(MatrixOut, grid_rows, grid_cols, ofile);
  // Write final output to output file
  if (write_out) {
    writeoutput(MatrixOut, grid_rows, grid_cols, ofile);
  }
#endif
  clReleaseMemObject(MatrixTemp[0]);
  clReleaseMemObject(MatrixTemp[1]);
  clReleaseMemObject(MatrixPower);

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
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);

  clReleaseContext(context);

  return 0;
}
