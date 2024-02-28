#include "./../main.h" // (in the main program folder)	needed to recognized input parameters
#include <CL/cl.h> // (in library path provided to compiler)	needed by OpenCL types and functions
#include <string.h>

#include "../../common/opencl_util.h" // (in directory)
#include "./../util/opencl/opencl.h" // (in library path specified to compiler)	needed by for device functions
#include "./../util/opencl/opencl.h"     // (in directory)
#include "./kernel_gpu_opencl_wrapper.h" // (in the current directory)
cl_context context;
cl_int error;
cl_command_queue command_queue;
cl_kernel kernel;
cl_program program;

void clInit() {

  // Get the number of available platforms
  cl_uint num_platforms;
  char pbuf[100];
  cl_platform_id *platform = NULL;
  cl_context_properties context_properties[3];
  cl_device_type device_type;
  display_device_info(&platform, &num_platforms);
  select_device_type(platform, &num_platforms, &device_type);
  validate_selection(platform, &num_platforms, context_properties,
                     &device_type);

  // Create context for selected platform being GPU
  context = clCreateContextFromType(context_properties, device_type, NULL, NULL,
                                    &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  // Get the number of devices (previousely selected for the context)
  size_t devices_size;
  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  // Get the list of devices (previousely selected for the context)
  cl_device_id *devices = (cl_device_id *)malloc(devices_size);
  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, devices,
                           NULL);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  // Select the first device (previousely selected for the context) (if there
  // are multiple devices, choose the first one)
  cl_device_id device;
  device = devices[0];

  // Get the name of the selected device (previousely selected for the context)
  // and print it
  error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(pbuf), pbuf, NULL);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
  printf("Device: %s\n", pbuf);
#endif
  // Create a command queue
  command_queue = clCreateCommandQueue(context, device, 0, &error);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  // Load kernel source code from file
  size_t sourceSize = 0;
  char *kernel_file_path = getVersionedKernelName("./kernel/lavaMD_kernel", 0);
  char *source = read_kernel(kernel_file_path, &sourceSize);
  free(kernel_file_path);

  program =
      clCreateProgramWithBinary(context, 1, devices, &sourceSize,
                                (const unsigned char **)&source, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  free(source);

#if defined(USE_JIT)
  // parameterized kernel dimension
  char clOptions[110];
  //  sprintf(clOptions,"-I../../src");
  sprintf(clOptions, "-I.");
#ifdef RD_WG_SIZE
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE=%d", RD_WG_SIZE);
#endif
#ifdef RD_WG_SIZE_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0=%d", RD_WG_SIZE_0);
#endif
#ifdef RD_WG_SIZE_0_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0_0=%d",
          RD_WG_SIZE_0_0);
#endif

  // Compile the program
  clBuildProgram_SAFE(program, 1, &device, clOptions, NULL, NULL);

#endif

  // Create kernel
  kernel = clCreateKernel(program, "kernel_gpu_opencl", &error);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
}
void kernel_gpu_opencl_wrapper(par_str par_cpu, dim_str dim_cpu,
                               box_str *box_cpu, FOUR_VECTOR *rv_cpu,
                               fp *qv_cpu, FOUR_VECTOR *fv_cpu, int version) {
  size_t local_work_size[1];
  local_work_size[0] = NUMBER_THREADS;
  size_t global_work_size[1];
  global_work_size[0] = dim_cpu.number_boxes * local_work_size[0];

  cl_mem d_par_gpu;
  d_par_gpu =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(par_cpu), NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_dim_gpu;
  d_dim_gpu =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(dim_cpu), NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_box_gpu;
  d_box_gpu =
      clCreateBuffer(context, CL_MEM_READ_WRITE, dim_cpu.box_mem, NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_rv_gpu;
  d_rv_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, dim_cpu.space_mem, NULL,
                            &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_qv_gpu;
  d_qv_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, dim_cpu.space_mem2,
                            NULL, &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  cl_mem d_fv_gpu;
  d_fv_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, dim_cpu.space_mem, NULL,
                            &error);
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);

  error = clEnqueueWriteBuffer(
      command_queue, // command queue
      d_par_gpu,     // destination
      1, // block the source from access until this copy operation complates
         // (1=yes, 0=no)
      0, // offset in destination to write to
      sizeof(par_cpu), // size to be copied
      &par_cpu,        // source
      0,               // # of events in the list of events to wait for
      NULL,            // list of events to wait for
      NULL);           // ID of this operation to be used by waiting operations

#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  error = clEnqueueWriteBuffer(
      command_queue, // command queue
      d_dim_gpu,     // destination
      1, // block the source from access until this copy operation complates
         // (1=yes, 0=no)
      0, // offset in destination to write to
      sizeof(dim_cpu), // size to be copied
      &dim_cpu,        // source
      0,               // # of events in the list of events to wait for
      NULL,            // list of events to wait for
      NULL);           // ID of this operation to be used by waiting operations
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  error = clEnqueueWriteBuffer(
      command_queue, // command queue
      d_box_gpu,     // destination
      1, // block the source from access until this copy operation complates
         // (1=yes, 0=no)
      0, // offset in destination to write to
      dim_cpu.box_mem, // size to be copied
      box_cpu,         // source
      0,               // # of events in the list of events to wait for
      NULL,            // list of events to wait for
      NULL);           // ID of this operation to be used by waiting operations
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  error = clEnqueueWriteBuffer(command_queue, d_rv_gpu, 1, 0, dim_cpu.space_mem,
                               rv_cpu, 0, 0, 0);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  error = clEnqueueWriteBuffer(command_queue, d_qv_gpu, 1, 0,
                               dim_cpu.space_mem2, qv_cpu, 0, 0, 0);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  error = clEnqueueWriteBuffer(command_queue, d_fv_gpu, 1, 0, dim_cpu.space_mem,
                               fv_cpu, 0, 0, 0);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  // kernel arguments
  CL_SAFE_CALL(
      clSetKernelArg(kernel, 0, sizeof(float), (void *)&par_cpu.alpha));
  CL_SAFE_CALL(
      clSetKernelArg(kernel, 1, sizeof(long), (void *)&dim_cpu.number_boxes));
  CL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_box_gpu));
  CL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_rv_gpu));
  CL_SAFE_CALL(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_qv_gpu));
  CL_SAFE_CALL(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_fv_gpu));

  // launch kernel - all boxes
  if (is_ndrange_kernel(version)) {
    error =
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size,
                               local_work_size, 0, NULL, NULL);
  } else {
    error = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);
  }
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
    // Wait for all operations to finish
//  error = clFinish(command_queue);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  error = clEnqueueReadBuffer(
      command_queue, // The command queue.
      d_fv_gpu,      // The image on the device.
      CL_TRUE, // Blocking? (ie. Wait at this line until read has finished?)
      0,       // Offset. None in this case.
      dim_cpu.space_mem, // Size to copy.
      fv_cpu,            // The pointer to the image on the host.
      0,                 // Number of events in wait list. Not used.
      NULL,              // Event wait list. Not used.
      NULL);             // Event object for determining status. Not used.
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
  // Clean up the device memory...
  clReleaseMemObject(d_rv_gpu);
  clReleaseMemObject(d_qv_gpu);
  clReleaseMemObject(d_fv_gpu);
  clReleaseMemObject(d_box_gpu);

  // Flush the queue
  error = clFlush(command_queue);
#ifdef DEBUG
  if (error != CL_SUCCESS)
    fatal_CL(error, __LINE__);
#endif
}

void clRelease() {
  clReleaseKernel(kernel);

  clReleaseProgram(program);

  clReleaseContext(context);

  clReleaseCommandQueue(command_queue);
}
