/****************************************************************************\
 * Copyright (c) 2011, Advanced Micro Devices, Inc.                           *
 * All rights reserved.                                                       *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * Redistributions of source code must retain the above copyright notice,     *
 * this list of conditions and the following disclaimer.                      *
 *                                                                            *
 * Redistributions in binary form must reproduce the above copyright notice,  *
 * this list of conditions and the following disclaimer in the documentation  *
 * and/or other materials provided with the distribution.                     *
 *                                                                            *
 * Neither the name of the copyright holder nor the names of its contributors *
 * may be used to endorse or promote products derived from this software      *
 * without specific prior written permission.                                 *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR *
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               *
 *                                                                            *
 * If you use the software (in whole or in part), you shall adhere to all     *
 * applicable U.S., European, and other export laws, including but not        *
 * limited to the U.S. Export Administration Regulations (“EAR”), (15 C.F.R.  *
 * Sections 730 through 774), and E.U. Council Regulation (EC) No 1334/2000   *
 * of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR, you       *
 * hereby certify that, except pursuant to a license granted by the United    *
 * States Department of Commerce Bureau of Industry and Security or as        *
 * otherwise permitted pursuant to a License Exception under the U.S. Export  *
 * Administration Regulations ("EAR"), you will not (1) export, re-export or  *
 * release to a national of a country in Country Groups D:1, E:1 or E:2 any   *
 * restricted technology, software, or source code you receive hereunder,     *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such *
 * technology or software, if such foreign produced direct product is subject *
 * to national security controls as identified on the Commerce Control List   *
 *(currently found in Supplement 1 to Part 774 of EAR).  For the most current *
 * Country Group listings, or for additional information about the EAR or     *
 * your obligations under those regulations, please refer to the U.S. Bureau  *
 * of Industry and Security’s website at http://www.bis.doc.gov/.             *
 \****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

#include "clutils.h"
#include "utils.h"

#include "../common/opencl_util.h"

// The following variables have file scope to simplify
// the utility functions

//! All discoverable OpenCL platforms
static cl_platform_id *platforms = NULL;
// static cl_uint numPlatforms;

//! All discoverable OpenCL devices (one pointer per platform)
static cl_device_id *devices = NULL;
static cl_uint numDevices;

//! The chosen OpenCL platform
// static cl_platform_id platform = NULL;

//! The chosen OpenCL device
static cl_device_id device = NULL;

//! OpenCL context
static cl_context context = NULL;

//! OpenCL command queue
static cl_command_queue commandQueue = NULL;
// static cl_command_queue commandQueueProf = NULL;
// static cl_command_queue commandQueueNoProf = NULL;

//! Global status of events
// static bool eventsEnabled = false;

//-------------------------------------------------------
//          Initialization and Cleanup
//-------------------------------------------------------

//! Initialize OpenCl environment on one device
/*!
    Init function for one device. Looks for supported devices and creates a
   context \return returns a context initialized
*/
cl_context cl_init_context(int platform, int dev, int quiet) {

  //	int printInfo=1;
  //	if (platform >= 0 && dev >= 0) printInfo = 0;

  cl_int status;

  size_t size;
  cl_platform_id *platforms = NULL;
  cl_context_properties context_properties[3];
  cl_device_type device_type;
  cl_uint num_platforms;

  display_device_info(&platforms, &num_platforms);
  select_device_type(platforms, &num_platforms, &device_type);
  validate_selection(platforms, &num_platforms, context_properties,
                     &device_type);

  context = clCreateContextFromType(context_properties, device_type, NULL, NULL,
                                    &status);
  if (cl_errChk(status, "creating Context", true)) {
    exit(1);
  }

  CL_SAFE_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size));
  numDevices = (int)(size / sizeof(cl_device_id));
  devices = (cl_device_id *)malloc(size);
  if (devices == NULL) {
    fprintf(stderr, "Failed to allocate memory for devices.\n");
    exit(1);
  }
  CL_SAFE_CALL(
      clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL));

  commandQueue = clCreateCommandQueue(context, devices[0],
                                      CL_QUEUE_PROFILING_ENABLE, &status);

  if (cl_errChk(status, "creating command queue", true)) {
    exit(1);
  }
  return context;
}
/*!
    Release all resources that the user doesn't have access to.
*/
void cl_cleanup() {
  // Free the command queue
  if (commandQueue) {
    clReleaseCommandQueue(commandQueue);
  }

  // Free the context
  if (context) {
    clReleaseContext(context);
  }

  free(devices);
  // free(numDevices);

  // Free the platforms
  free(platforms);
}

//! Release a kernel object
/*!
    \param mem The kernel object to release
*/
void cl_freeKernel(cl_kernel kernel) {
  cl_int status;

  if (kernel != NULL) {
    status = clReleaseKernel(kernel);
    cl_errChk(status, "Releasing kernel object", true);
  }
}

//! Release memory allocated on the device
/*!
    \param mem The device pointer to release
*/
void cl_freeMem(cl_mem mem) {
  cl_int status;

  if (mem != NULL) {
    status = clReleaseMemObject(mem);
    cl_errChk(status, "Releasing mem object", true);
  }
}

//! Release a program object
/*!
    \param mem The program object to release
*/
void cl_freeProgram(cl_program program) {
  cl_int status;

  if (program != NULL) {
    status = clReleaseProgram(program);
    cl_errChk(status, "Releasing program object", true);
  }
}

//! Returns a reference to the command queue
/*!
        Returns a reference to the command queue \n
        Used for any OpenCl call that needs the command queue declared in
   clutils.cpp
*/
cl_command_queue cl_getCommandQueue() { return commandQueue; }

//-------------------------------------------------------
//          Synchronization functions
//-------------------------------------------------------

/*!
    Wait till all pending commands in queue are finished
*/
void cl_sync() { clFinish(commandQueue); }

//-------------------------------------------------------
//          Memory allocation
//-------------------------------------------------------

//! Allocate a buffer on a device
/*!
    \param mem_size Size of memory in bytes
    \param flags Optional cl_mem_flags
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocBuffer(size_t mem_size, cl_mem_flags flags) {
  cl_mem mem;
  cl_int status;

  /*!
      Logging information for keeping track of device memory
  */
  static int allocationCount = 1;
  static size_t allocationSize = 0;

  allocationCount++;
  allocationSize += mem_size;

  mem = clCreateBuffer(context, flags, mem_size, NULL, &status);

  cl_errChk(status, "creating buffer", true);

  return mem;
}

//! Allocate constant memory on device
/*!
    \param mem_size Size of memory in bytes
    \param host_ptr Host pointer that contains the data
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocBufferConst(size_t mem_size, void *host_ptr) {
  cl_mem mem;
  cl_int status;

  mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       mem_size, host_ptr, &status);
  cl_errChk(status, "Error creating const mem buffer", true);

  return mem;
}

//! Allocate a buffer on device pinning the host memory at host_ptr
/*!
    \param mem_size Size of memory in bytes
    \return Returns a cl_mem object that points to pinned memory on the host
*/
cl_mem cl_allocBufferPinned(size_t mem_size) {
  cl_mem mem;
  cl_int status;

  mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       mem_size, NULL, &status);
  cl_errChk(status, "Error allocating pinned memory", true);

  return mem;
}

//! Allocate an image on a device
/*!
    \param height Number of rows in the image
    \param width Number of columns in the image
    \param elemSize Size of the elements in the image
    \param flags Optional cl_mem_flags
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocImage(size_t height, size_t width, char type,
                     cl_mem_flags flags) {
  cl_mem mem;
  cl_int status;

  size_t elemSize = 0;

  cl_image_format format;
  format.image_channel_order = CL_R;

  switch (type) {
  case 'f':
    elemSize = sizeof(float);
    format.image_channel_data_type = CL_FLOAT;
    break;
  case 'i':
    elemSize = sizeof(int);
    format.image_channel_data_type = CL_SIGNED_INT32;
    break;
  default:
    printf("Error creating image: Unsupported image type.\n");
    exit(-1);
  }

  /*!
      Logging information for keeping track of device memory
  */
  static int allocationCount = 1;
  static size_t allocationSize = 0;

  allocationCount++;
  allocationSize += height * width * elemSize;

  // Create the image
  mem =
      clCreateImage2D(context, flags, &format, width, height, 0, NULL, &status);

  // cl_errChk(status, "creating image", true);
  if (status != CL_SUCCESS) {
    printf(
        "Error creating image: Images may not be supported for this device.\n");
    printSupportedImageFormats();
    getchar();
    exit(-1);
  }

  return mem;
}

//-------------------------------------------------------
//          Data transfers
//-------------------------------------------------------

// Copy and map a buffer
void *cl_copyAndMapBuffer(cl_mem dst, cl_mem src, size_t size) {

  void *ptr; // Pointer to the pinned memory that will be returned

  cl_copyBufferToBuffer(dst, src, size);

  ptr = cl_mapBuffer(dst, size, CL_MAP_READ);

  return ptr;
}

// Copy a buffer
void cl_copyBufferToBuffer(cl_mem dst, cl_mem src, size_t size) {
  cl_int status;
  status =
      clEnqueueCopyBuffer(commandQueue, src, dst, 0, 0, size, 0, NULL, NULL);
  cl_errChk(status, "Copying buffer", true);
}

//! Copy a buffer to the device
/*!
    \param dst Valid device pointer
    \param src Host pointer that contains the data
    \param mem_size Size of data to copy
        \param blocking Blocking or non-blocking operation
*/
void cl_copyBufferToDevice(cl_mem dst, void *src, size_t mem_size,
                           cl_bool blocking) {
  cl_int status;
  status = clEnqueueWriteBuffer(commandQueue, dst, blocking, 0, mem_size, src,
                                0, NULL, NULL);
  cl_errChk(status, "Writing buffer", true);
}

//! Copy a buffer to the host
/*!
    \param dst Valid host pointer
    \param src Device pointer that contains the data
    \param mem_size Size of data to copy
        \param blocking Blocking or non-blocking operation
*/
void cl_copyBufferToHost(void *dst, cl_mem src, size_t mem_size,
                         cl_bool blocking) {
  cl_int status;
  status = clEnqueueReadBuffer(commandQueue, src, blocking, 0, mem_size, dst, 0,
                               NULL, NULL);
  cl_errChk(status, "Reading buffer", true);
}

//! Copy a buffer to a 2D image
/*!
    \param src Valid device buffer
    \param dst Empty device image
    \param mem_size Size of data to copy
*/
void cl_copyBufferToImage(cl_mem buffer, cl_mem image, int height, int width) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {(size_t)width, (size_t)height, 1};

  cl_int status;
  status = clEnqueueCopyBufferToImage(commandQueue, buffer, image, 0, origin,
                                      region, 0, NULL, NULL);
  cl_errChk(status, "Copying buffer to image", true);
}

// Copy data to an image on the device
/*!
    \param dst Valid device pointer
    \param src Host pointer that contains the data
    \param height Height of the image
    \param width Width of the image
*/
void cl_copyImageToDevice(cl_mem dst, void *src, size_t height, size_t width) {
  cl_int status;
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};

  status = clEnqueueWriteImage(commandQueue, dst, CL_TRUE, origin, region, 0, 0,
                               src, 0, NULL, NULL);
  cl_errChk(status, "Writing image", true);
}

//! Copy an image to the host
/*!
    \param dst Valid host pointer
    \param src Device pointer that contains the data
    \param height Height of the image
    \param width Width of the image
*/
void cl_copyImageToHost(void *dst, cl_mem src, size_t height, size_t width) {
  cl_int status;
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};

  status = clEnqueueReadImage(commandQueue, src, CL_TRUE, origin, region, 0, 0,
                              dst, 0, NULL, NULL);
  cl_errChk(status, "Reading image", true);
}

//! Map a buffer into a host address
/*!
    \param mem cl_mem object
        \param mem_size Size of memory in bytes
    \param flags Optional cl_mem_flags
    \return Returns a host pointer that points to the mapped region
*/
void *cl_mapBuffer(cl_mem mem, size_t mem_size, cl_mem_flags flags) {
  cl_int status;
  void *ptr;

  ptr = (void *)clEnqueueMapBuffer(commandQueue, mem, CL_TRUE, flags, 0,
                                   mem_size, 0, NULL, NULL, &status);

  cl_errChk(status, "Error mapping a buffer", true);

  return ptr;
}

//! Unmap a buffer or image
/*!
    \param mem cl_mem object
    \param ptr A host pointer that points to the mapped region
*/
void cl_unmapBuffer(cl_mem mem, void *ptr) {

  // TODO It looks like AMD doesn't support profiling unmapping yet. Leaving the
  //      commented code here until it's supported

  cl_int status;

  status = clEnqueueUnmapMemObject(commandQueue, mem, ptr, 0, NULL, NULL);

  cl_errChk(status, "Error unmapping a buffer or image", true);
}

void cl_writeToZCBuffer(cl_mem mem, void *data, size_t size) {

  void *ptr;

  ptr = cl_mapBuffer(mem, size, CL_MAP_WRITE);

  memcpy(ptr, data, size);

  cl_unmapBuffer(mem, ptr);
}

//-------------------------------------------------------
//          Program and kernels
//-------------------------------------------------------

//! Convert source code file into cl_program
/*!
Compile Opencl source file into a cl_program. The cl_program will be made into a
kernel in PrecompileKernels()

\param kernelPath  Filename of OpenCl code
\param compileoptions Compilation options
\param verbosebuild Switch to enable verbose Output
*/
cl_program cl_compileProgram(char *kernelPath, char *compileoptions) {
  cl_int status;
  size_t size;

  char *source = read_kernel(kernelPath, &size);

  // Create the program object
  cl_program clProgramReturn =
      clCreateProgramWithBinary(context, 1, devices, &size,
                                (const unsigned char **)&source, NULL, &status);

  cl_errChk(status, "Creating program", true);

  free(source);

  // Try to compile the program
  clBuildProgram_SAFE(clProgramReturn, numDevices, devices, compileoptions,
                      NULL, NULL);

  return clProgramReturn;
}

//! Create a kernel from compiled source
/*!
Create a kernel from compiled source

\param program  Compiled OpenCL program
\param kernel_name  Name of the kernel in the program
\return Returns a cl_kernel object for the specified kernel
*/
cl_kernel cl_createKernel(cl_program program, const char *kernel_name) {

  cl_kernel kernel;
  cl_int status;

  kernel = clCreateKernel(program, kernel_name, &status);
  cl_errChk(status, "Creating kernel", true);

  return kernel;
}

//! Set an argument for a OpenCL kernel
/*!
Set an argument for a OpenCL kernel

\param kernel The kernel for which the argument is being set
\param index The argument index
\param size The size of the argument
\param data A pointer to the argument
*/
void cl_setKernelArg(cl_kernel kernel, unsigned int index, size_t size,
                     void *data) {
  cl_int status;
  status = clSetKernelArg(kernel, index, size, data);

  cl_errChk(status, "Setting kernel arg", true);
}

//-------------------------------------------------------
//          Profiling/events
//-------------------------------------------------------

//! Time kernel execution using cl_event
/*!
    Prints out the time taken between the start and end of an event
    \param event_time
*/
double cl_computeExecTime(cl_event event_time) {
  cl_int status;
  cl_ulong starttime;
  cl_ulong endtime;

  double elapsed;

  status = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &starttime, NULL);
  cl_errChk(status, "profiling start", true);

  status = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &endtime, NULL);
  cl_errChk(status, "profiling end", true);

  // Convert to ms
  elapsed = (double)(endtime - starttime) / 1000000.0;

  return elapsed;
}

//! Compute the elapsed time between two timer values
double cl_computeTime(cl_time start, cl_time end) {
#ifdef _WIN32
  __int64 freq;
  int status;

  status = QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
  if (status == 0) {
    perror("QueryPerformanceFrequency");
    exit(-1);
  }

  // Return time in ms
  return double(end - start) / (double(freq) / 1000.0);
#else

  return end - start;
#endif
}

//! Grab the current time using a system-specific timer
void cl_getTime(cl_time *time) {

#ifdef _WIN32
  int status = QueryPerformanceCounter((LARGE_INTEGER *)time);
  if (status == 0) {
    perror("QueryPerformanceCounter");
    exit(-1);
  }
#else
  // Use gettimeofday to get the current time
  struct timeval curTime;
  gettimeofday(&curTime, NULL);

  // Convert timeval into double
  *time = curTime.tv_sec * 1000 + (double)curTime.tv_usec / 1000;
#endif
}

//-------------------------------------------------------
//          Error handling
//-------------------------------------------------------

//! OpenCl error code list
/*!
    An array of character strings used to give the error corresponding to the
   error code \n

    The error code is the index within this array
*/
char *cl_errs[MAX_ERR_VAL] = {
    (char *)"CL_SUCCESS",                         // 0
    (char *)"CL_DEVICE_NOT_FOUND",                //-1
    (char *)"CL_DEVICE_NOT_AVAILABLE",            //-2
    (char *)"CL_COMPILER_NOT_AVAILABLE",          //-3
    (char *)"CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4
    (char *)"CL_OUT_OF_RESOURCES",                //-5
    (char *)"CL_OUT_OF_HOST_MEMORY",              //-6
    (char *)"CL_PROFILING_INFO_NOT_AVAILABLE",    //-7
    (char *)"CL_MEM_COPY_OVERLAP",                //-8
    (char *)"CL_IMAGE_FORMAT_MISMATCH",           //-9
    (char *)"CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
    (char *)"CL_BUILD_PROGRAM_FAILURE",           //-11
    (char *)"CL_MAP_FAILURE",                     //-12
    (char *)"",                                   //-13
    (char *)"",                                   //-14
    (char *)"",                                   //-15
    (char *)"",                                   //-16
    (char *)"",                                   //-17
    (char *)"",                                   //-18
    (char *)"",                                   //-19
    (char *)"",                                   //-20
    (char *)"",                                   //-21
    (char *)"",                                   //-22
    (char *)"",                                   //-23
    (char *)"",                                   //-24
    (char *)"",                                   //-25
    (char *)"",                                   //-26
    (char *)"",                                   //-27
    (char *)"",                                   //-28
    (char *)"",                                   //-29
    (char *)"CL_INVALID_VALUE",                   //-30
    (char *)"CL_INVALID_DEVICE_TYPE",             //-31
    (char *)"CL_INVALID_PLATFORM",                //-32
    (char *)"CL_INVALID_DEVICE",                  //-33
    (char *)"CL_INVALID_CONTEXT",                 //-34
    (char *)"CL_INVALID_QUEUE_PROPERTIES",        //-35
    (char *)"CL_INVALID_COMMAND_QUEUE",           //-36
    (char *)"CL_INVALID_HOST_PTR",                //-37
    (char *)"CL_INVALID_MEM_OBJECT",              //-38
    (char *)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
    (char *)"CL_INVALID_IMAGE_SIZE",              //-40
    (char *)"CL_INVALID_SAMPLER",                 //-41
    (char *)"CL_INVALID_BINARY",                  //-42
    (char *)"CL_INVALID_BUILD_OPTIONS",           //-43
    (char *)"CL_INVALID_PROGRAM",                 //-44
    (char *)"CL_INVALID_PROGRAM_EXECUTABLE",      //-45
    (char *)"CL_INVALID_KERNEL_NAME",             //-46
    (char *)"CL_INVALID_KERNEL_DEFINITION",       //-47
    (char *)"CL_INVALID_KERNEL",                  //-48
    (char *)"CL_INVALID_ARG_INDEX",               //-49
    (char *)"CL_INVALID_ARG_VALUE",               //-50
    (char *)"CL_INVALID_ARG_SIZE",                //-51
    (char *)"CL_INVALID_KERNEL_ARGS",             //-52
    (char *)"CL_INVALID_WORK_DIMENSION ",         //-53
    (char *)"CL_INVALID_WORK_GROUP_SIZE",         //-54
    (char *)"CL_INVALID_WORK_ITEM_SIZE",          //-55
    (char *)"CL_INVALID_GLOBAL_OFFSET",           //-56
    (char *)"CL_INVALID_EVENT_WAIT_LIST",         //-57
    (char *)"CL_INVALID_EVENT",                   //-58
    (char *)"CL_INVALID_OPERATION",               //-59
    (char *)"CL_INVALID_GL_OBJECT",               //-60
    (char *)"CL_INVALID_BUFFER_SIZE",             //-61
    (char *)"CL_INVALID_MIP_LEVEL",               //-62
    (char *)"CL_INVALID_GLOBAL_WORK_SIZE"};       //-63

//! OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl
\param status Error value as cl_int
\param msg User provided error message
\return True if Error Seen, False if no error
*/
int cl_errChk(const cl_int status, const char *msg, bool exitOnErr) {
#ifdef DEBUG
  if (status != CL_SUCCESS) {
    printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);

    if (exitOnErr) {
      exit(-1);
    }

    return true;
  }
#endif
  return false;
}

// Queries the supported image formats for the device and prints
// them to the screen
void printSupportedImageFormats() {
  cl_uint numFormats;
  cl_int status;

  status = clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D, 0,
                                      NULL, &numFormats);
  cl_errChk(status, "getting supported image formats", true);

  cl_image_format *imageFormats = NULL;
  imageFormats = (cl_image_format *)alloc(sizeof(cl_image_format) * numFormats);

  status = clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D,
                                      numFormats, imageFormats, NULL);

  printf("There are %d supported image formats\n", numFormats);

  cl_uint orders[] = {CL_R,  CL_A,   CL_INTENSITY, CL_LUMINANCE, CL_RG,
                      CL_RA, CL_RGB, CL_RGBA,      CL_ARGB,      CL_BGRA};
  char *orderstr[] = {(char *)"CL_R",         (char *)"CL_A",
                      (char *)"CL_INTENSITY", (char *)"CL_LUMINANCE",
                      (char *)"CL_RG",        (char *)"CL_RA",
                      (char *)"CL_RGB",       (char *)"CL_RGBA",
                      (char *)"CL_ARGB",      (char *)"CL_BGRA"};

  cl_uint types[] = {
      CL_SNORM_INT8,       CL_SNORM_INT16,     CL_UNORM_INT8,
      CL_UNORM_INT16,      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555,
      CL_UNORM_INT_101010, CL_SIGNED_INT8,     CL_SIGNED_INT16,
      CL_SIGNED_INT32,     CL_UNSIGNED_INT8,   CL_UNSIGNED_INT16,
      CL_UNSIGNED_INT32,   CL_HALF_FLOAT,      CL_FLOAT};

  char *typesstr[] = {(char *)"CL_SNORM_INT8",
                      (char *)"CL_SNORM_INT16",
                      (char *)"CL_UNORM_INT8",
                      (char *)"CL_UNORM_INT16",
                      (char *)"CL_UNORM_SHORT_565",
                      (char *)"CL_UNORM_SHORT_555",
                      (char *)"CL_UNORM_INT_101010",
                      (char *)"CL_SIGNED_INT8",
                      (char *)"CL_SIGNED_INT16",
                      (char *)"CL_SIGNED_INT32",
                      (char *)"CL_UNSIGNED_INT8",
                      (char *)"CL_UNSIGNED_INT16",
                      (char *)"CL_UNSIGNED_INT32",
                      (char *)"CL_HALF_FLOAT",
                      (char *)"CL_FLOAT"};

  printf("Supported Formats:\n");
  for (int i = 0; i < (int)numFormats; i++) {
    printf("\tFormat %d: ", i);

    for (int j = 0; j < (int)(sizeof(orders) / sizeof(cl_int)); j++) {
      if (imageFormats[i].image_channel_order == orders[j]) {
        printf("%s, ", orderstr[j]);
      }
    }
    for (int j = 0; j < (int)(sizeof(types) / sizeof(cl_int)); j++) {
      if (imageFormats[i].image_channel_data_type == types[j]) {
        printf("%s, ", typesstr[j]);
      }
    }
    printf("\n");
  }

  free(imageFormats);
}

//-------------------------------------------------------
//          Platform and device information
//-------------------------------------------------------

//! Returns true if AMD is the device vendor
bool cl_deviceIsAMD(cl_device_id dev) {

  bool retval = false;

  char *vendor = cl_getDeviceVendor(dev);

  if (strncmp(vendor, "Advanced", 8) == 0) {
    retval = true;
  }

  free(vendor);

  return retval;
}

//! Returns true if NVIDIA is the device vendor
bool cl_deviceIsNVIDIA(cl_device_id dev) {

  bool retval = false;

  char *vendor = cl_getDeviceVendor(dev);

  if (strncmp(vendor, "NVIDIA", 6) == 0) {
    retval = true;
  }

  free(vendor);

  return retval;
}

//! Returns true if NVIDIA is the device vendor
bool cl_platformIsNVIDIA(cl_platform_id plat) {

  bool retval = false;

  char *vendor = cl_getPlatformVendor(plat);

  if (strncmp(vendor, "NVIDIA", 6) == 0) {
    retval = true;
  }

  free(vendor);

  return retval;
}

//! Get the name of the vendor for a device
char *cl_getDeviceDriverVersion(cl_device_id dev) {
  cl_int status;
  size_t devInfoSize;
  char *devInfoStr = NULL;

  // If dev is NULL, set it to the default device
  if (dev == NULL) {
    dev = device;
  }

  // Print the vendor
  status = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &devInfoSize);
  cl_errChk(status, "Getting vendor name", true);

  devInfoStr = (char *)alloc(devInfoSize);

  status =
      clGetDeviceInfo(dev, CL_DRIVER_VERSION, devInfoSize, devInfoStr, NULL);
  cl_errChk(status, "Getting vendor name", true);

  return devInfoStr;
}

//! The the name of the device as supplied by the OpenCL implementation
char *cl_getDeviceName(cl_device_id dev) {
  cl_int status;
  size_t devInfoSize;
  char *devInfoStr = NULL;

  // If dev is NULL, set it to the default device
  if (dev == NULL) {
    dev = device;
  }

  // Print the name
  status = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &devInfoSize);
  cl_errChk(status, "Getting device name", true);

  devInfoStr = (char *)alloc(devInfoSize);

  status = clGetDeviceInfo(dev, CL_DEVICE_NAME, devInfoSize, devInfoStr, NULL);
  cl_errChk(status, "Getting device name", true);

  return (devInfoStr);
}

//! Get the name of the vendor for a device
char *cl_getDeviceVendor(cl_device_id dev) {
  cl_int status;
  size_t devInfoSize;
  char *devInfoStr = NULL;

  // If dev is NULL, set it to the default device
  if (dev == NULL) {
    dev = device;
  }

  // Print the vendor
  status = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, NULL, &devInfoSize);
  cl_errChk(status, "Getting vendor name", true);

  devInfoStr = (char *)alloc(devInfoSize);

  status =
      clGetDeviceInfo(dev, CL_DEVICE_VENDOR, devInfoSize, devInfoStr, NULL);
  cl_errChk(status, "Getting vendor name", true);

  return devInfoStr;
}

//! Get the name of the vendor for a device
char *cl_getDeviceVersion(cl_device_id dev) {
  cl_int status;
  size_t devInfoSize;
  char *devInfoStr = NULL;

  // If dev is NULL, set it to the default device
  if (dev == NULL) {
    dev = device;
  }

  // Print the vendor
  status = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, NULL, &devInfoSize);
  cl_errChk(status, "Getting vendor name", true);

  devInfoStr = (char *)alloc(devInfoSize);

  status =
      clGetDeviceInfo(dev, CL_DEVICE_VERSION, devInfoSize, devInfoStr, NULL);
  cl_errChk(status, "Getting vendor name", true);

  return devInfoStr;
}

//! The the name of the device as supplied by the OpenCL implementation
char *cl_getPlatformName(cl_platform_id platform) {
  cl_int status;
  size_t platformInfoSize;
  char *platformInfoStr = NULL;

  // Print the name
  status =
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &platformInfoSize);
  cl_errChk(status, "Getting platform name", true);

  platformInfoStr = (char *)alloc(platformInfoSize);

  status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformInfoSize,
                             platformInfoStr, NULL);
  cl_errChk(status, "Getting platform name", true);

  return (platformInfoStr);
}

//! The the name of the device as supplied by the OpenCL implementation
char *cl_getPlatformVendor(cl_platform_id platform) {
  cl_int status;
  size_t platformInfoSize;
  char *platformInfoStr = NULL;

  // Print the name
  status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL,
                             &platformInfoSize);
  cl_errChk(status, "Getting platform name", true);

  platformInfoStr = (char *)alloc(platformInfoSize);

  status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformInfoSize,
                             platformInfoStr, NULL);
  cl_errChk(status, "Getting platform name", true);

  return (platformInfoStr);
}
