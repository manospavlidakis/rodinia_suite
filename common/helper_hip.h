/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_H
#include <hip/hip_runtime.h>

#define HELPER_CUDA_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_string.h"

/*
inline void __ExitInTime(int seconds)
{
    fprintf(stdout, "> exiting in %d seconds: ", seconds);
    fflush(stdout);
    time_t t;
    int count;

    for (t=time(0)+seconds, count=seconds; time(0) < t; count--) {
        fprintf(stdout, "%d...", count);
#if defined(WIN32)
        Sleep(1000);
#else
        sleep(1);
#endif
    }

    fprintf(stdout,"done!\n\n");
    fflush(stdout);
}

#define EXIT_TIME_DELAY 2

inline void EXIT_DELAY(int return_code)
{
    __ExitInTime(EXIT_TIME_DELAY);
    exit(return_code);
}
*/

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef DISABLE_HIP_CHECK
#define HIP_CHECK(call) (void)(call)
#else
#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t _e = (call);                                                  \
    if (_e != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP ERROR %s:%d: %s failed: %s\n",               \
                   __FILE__, __LINE__, #call, hipGetErrorString(_e));        \
      std::abort();                                                          \
    }                                                                        \
  } while (0)
#endif

// Note, it is required that your SDK sample to include the proper header files,
// please refer the CUDA examples for examples of the needed CUDA headers, which
// may change depending on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(hipError_t error) {
  switch (error) {
  case hipSuccess:
    return "hipSuccess";

  case hipErrorMissingConfiguration:
    return "hipErrorMissingConfiguration";

  case hipErrorOutOfMemory:
    return "hipErrorOutOfMemory";

  case hipErrorNotInitialized:
    return "hipErrorNotInitialized";

  case hipErrorLaunchFailure:
    return "hipErrorLaunchFailure";

  case hipErrorPriorLaunchFailure:
    return "hipErrorPriorLaunchFailure";

  case hipErrorLaunchTimeOut:
    return "hipErrorLaunchTimeOut";

  case hipErrorLaunchOutOfResources:
    return "hipErrorLaunchOutOfResources";

  case hipErrorInvalidDeviceFunction:
    return "hipErrorInvalidDeviceFunction";

  case hipErrorInvalidConfiguration:
    return "hipErrorInvalidConfiguration";

  case hipErrorInvalidDevice:
    return "hipErrorInvalidDevice";

  case hipErrorInvalidValue:
    return "hipErrorInvalidValue";

  case hipErrorInvalidPitchValue:
    return "hipErrorInvalidPitchValue";

  case hipErrorInvalidSymbol:
    return "hipErrorInvalidSymbol";

  case hipErrorMapFailed:
    return "hipErrorMapFailed";

  case hipErrorUnmapFailed:
    return "hipErrorUnmapFailed";

  case cudaErrorInvalidHostPointer:
    return "hipErrorInvalidHostPointer";

  case hipErrorInvalidDevicePointer:
    return "hipErrorInvalidDevicePointer";

  case cudaErrorInvalidTexture:
    return "hipErrorInvalidTexture";

  case cudaErrorInvalidTextureBinding:
    return "hipErrorInvalidTextureBinding";

  case cudaErrorInvalidChannelDescriptor:
    return "hipErrorInvalidChannelDescriptor";

  case hipErrorInvalidMemcpyDirection:
    return "hipErrorInvalidMemcpyDirection";

  case cudaErrorAddressOfConstant:
    return "hipErrorAddressOfConstant";

  case cudaErrorTextureFetchFailed:
    return "hipErrorTextureFetchFailed";

  case cudaErrorTextureNotBound:
    return "hipErrorTextureNotBound";

  case cudaErrorSynchronizationError:
    return "hipErrorSynchronizationError";

  case cudaErrorInvalidFilterSetting:
    return "hipErrorInvalidFilterSetting";

  case cudaErrorInvalidNormSetting:
    return "hipErrorInvalidNormSetting";

  case cudaErrorMixedDeviceExecution:
    return "hipErrorMixedDeviceExecution";

  case hipErrorDeinitialized:
    return "hipErrorDeinitialized";

  case hipErrorUnknown:
    return "hipErrorUnknown";

  case cudaErrorNotYetImplemented:
    return "hipErrorNotYetImplemented";

  case cudaErrorMemoryValueTooLarge:
    return "hipErrorMemoryValueTooLarge";

  case hipErrorInvalidHandle:
    return "hipErrorInvalidHandle";

  case hipErrorNotReady:
    return "hipErrorNotReady";

  case hipErrorInsufficientDriver:
    return "hipErrorInsufficientDriver";

  case hipErrorSetOnActiveProcess:
    return "hipErrorSetOnActiveProcess";

  case cudaErrorInvalidSurface:
    return "hipErrorInvalidSurface";

  case hipErrorNoDevice:
    return "hipErrorNoDevice";

  case hipErrorECCNotCorrectable:
    return "hipErrorECCNotCorrectable";

  case hipErrorSharedObjectSymbolNotFound:
    return "hipErrorSharedObjectSymbolNotFound";

  case hipErrorSharedObjectInitFailed:
    return "hipErrorSharedObjectInitFailed";

  case hipErrorUnsupportedLimit:
    return "hipErrorUnsupportedLimit";

  case cudaErrorDuplicateVariableName:
    return "hipErrorDuplicateVariableName";

  case cudaErrorDuplicateTextureName:
    return "hipErrorDuplicateTextureName";

  case cudaErrorDuplicateSurfaceName:
    return "hipErrorDuplicateSurfaceName";

  case cudaErrorDevicesUnavailable:
    return "hipErrorDevicesUnavailable";

  case hipErrorInvalidImage:
    return "hipErrorInvalidImage";

  case hipErrorNoBinaryForGpu:
    return "hipErrorNoBinaryForGpu";

  case cudaErrorIncompatibleDriverContext:
    return "hipErrorIncompatibleDriverContext";

  case hipErrorPeerAccessAlreadyEnabled:
    return "hipErrorPeerAccessAlreadyEnabled";

  case hipErrorPeerAccessNotEnabled:
    return "hipErrorPeerAccessNotEnabled";

  case hipErrorContextAlreadyInUse:
    return "hipErrorContextAlreadyInUse";

  case hipErrorProfilerDisabled:
    return "hipErrorProfilerDisabled";

  case hipErrorProfilerNotInitialized:
    return "hipErrorProfilerNotInitialized";

  case hipErrorProfilerAlreadyStarted:
    return "hipErrorProfilerAlreadyStarted";

  case hipErrorProfilerAlreadyStopped:
    return "hipErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

  case hipErrorAssert:
    return "hipErrorAssert";

  case cudaErrorTooManyPeers:
    return "hipErrorTooManyPeers";

  case hipErrorHostMemoryAlreadyRegistered:
    return "hipErrorHostMemoryAlreadyRegistered";

  case hipErrorHostMemoryNotRegistered:
    return "hipErrorHostMemoryNotRegistered";
#endif

  case cudaErrorStartupFailure:
    return "hipErrorStartupFailure";

  case cudaErrorApiFailureBase:
    return "hipErrorApiFailureBase";
  }

  return "<unknown>";
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(hipError_t error) {
  switch (error) {
  case hipSuccess:
    return "CUDA_SUCCESS";

  case hipErrorInvalidValue:
    return "CUDA_ERROR_INVALID_VALUE";

  case hipErrorOutOfMemory:
    return "CUDA_ERROR_OUT_OF_MEMORY";

  case hipErrorNotInitialized:
    return "CUDA_ERROR_NOT_INITIALIZED";

  case hipErrorDeinitialized:
    return "CUDA_ERROR_DEINITIALIZED";

  case hipErrorProfilerDisabled:
    return "CUDA_ERROR_PROFILER_DISABLED";

  case hipErrorProfilerNotInitialized:
    return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

  case hipErrorProfilerAlreadyStarted:
    return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

  case hipErrorProfilerAlreadyStopped:
    return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

  case hipErrorNoDevice:
    return "CUDA_ERROR_NO_DEVICE";

  case hipErrorInvalidDevice:
    return "CUDA_ERROR_INVALID_DEVICE";

  case hipErrorInvalidImage:
    return "CUDA_ERROR_INVALID_IMAGE";

  case hipErrorInvalidContext:
    return "CUDA_ERROR_INVALID_CONTEXT";

  case hipErrorContextAlreadyCurrent:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

  case hipErrorMapFailed:
    return "CUDA_ERROR_MAP_FAILED";

  case hipErrorUnmapFailed:
    return "CUDA_ERROR_UNMAP_FAILED";

  case hipErrorArrayIsMapped:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";

  case hipErrorAlreadyMapped:
    return "CUDA_ERROR_ALREADY_MAPPED";

  case hipErrorNoBinaryForGpu:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";

  case hipErrorAlreadyAcquired:
    return "CUDA_ERROR_ALREADY_ACQUIRED";

  case hipErrorNotMapped:
    return "CUDA_ERROR_NOT_MAPPED";

  case hipErrorNotMappedAsArray:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

  case hipErrorNotMappedAsPointer:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

  case hipErrorECCNotCorrectable:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";

  case hipErrorUnsupportedLimit:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";

  case hipErrorContextAlreadyInUse:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

  case hipErrorInvalidSource:
    return "CUDA_ERROR_INVALID_SOURCE";

  case hipErrorFileNotFound:
    return "CUDA_ERROR_FILE_NOT_FOUND";

  case hipErrorSharedObjectSymbolNotFound:
    return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

  case hipErrorSharedObjectInitFailed:
    return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

  case hipErrorOperatingSystem:
    return "CUDA_ERROR_OPERATING_SYSTEM";

  case hipErrorInvalidHandle:
    return "CUDA_ERROR_INVALID_HANDLE";

  case hipErrorNotFound:
    return "CUDA_ERROR_NOT_FOUND";

  case hipErrorNotReady:
    return "CUDA_ERROR_NOT_READY";

  case hipErrorLaunchFailure:
    return "CUDA_ERROR_LAUNCH_FAILED";

  case hipErrorLaunchOutOfResources:
    return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

  case hipErrorLaunchTimeOut:
    return "CUDA_ERROR_LAUNCH_TIMEOUT";

  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
    return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

  case hipErrorPeerAccessAlreadyEnabled:
    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

  case hipErrorPeerAccessNotEnabled:
    return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

  case hipErrorSetOnActiveProcess:
    return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

  case hipErrorContextIsDestroyed:
    return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

  case hipErrorAssert:
    return "CUDA_ERROR_ASSERT";

  case CUDA_ERROR_TOO_MANY_PEERS:
    return "CUDA_ERROR_TOO_MANY_PEERS";

  case hipErrorHostMemoryAlreadyRegistered:
    return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

  case hipErrorHostMemoryNotRegistered:
    return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

  case hipErrorUnknown:
    return "CUDA_ERROR_UNKNOWN";
  }

  return "<unknown>";
}
#endif
#define __DRIVER_TYPES_H__
#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET HIP_CHECK(hipDeviceReset());
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  hipError_t err = hipGetLastError();

  if (hipSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, hipGetErrorString(err));
    DEVICE_RESET
    exit(EXIT_FAILURE);
  }
}

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the #
  // of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
            // minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x10, 8},   // Tesla Generation (SM 1.0) G80 class
      {0x11, 8},   // Tesla Generation (SM 1.1) G8x class
      {0x12, 8},   // Tesla Generation (SM 1.2) G9x class
      {0x13, 8},   // Tesla Generation (SM 1.3) GT200 class
      {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
      {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
      {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
      {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run
  // properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[7].Cores);
  return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  checkCudaErrors(hipGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  hipDeviceProp_t deviceProp;
  checkCudaErrors(hipGetDeviceProperties(&deviceProp, devID));

  if (deviceProp.computeMode == hipComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no "
                    "threads can use ::cudaSetDevice().\n");
    return -1;
  }

  if (deviceProp.major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(hipSetDevice(devID));
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;

  unsigned long long max_compute_perf = 0;
  hipDeviceProp_t deviceProp;
  hipGetDeviceCount(&device_count);

  checkCudaErrors(hipGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(
        stderr,
        "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    hipGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited, then we can add it
    // to the list
    if (deviceProp.computeMode != hipComputeModeProhibited) {
      if (deviceProp.major > 0 && deviceProp.major < 9999) {
        best_SM_arch = MAX(best_SM_arch, deviceProp.major);
      }
    }

    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    hipGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited, then we can add it
    // to the list
    if (deviceProp.computeMode != hipComputeModeProhibited) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
      }

      unsigned long long compute_perf =
          (unsigned long long)deviceProp.multiProcessorCount *
          sm_per_multiproc * deviceProp.clockRate;

      if (compute_perf > max_compute_perf) {
        // If we find GPU with SM major > 2, search only these
        if (best_SM_arch > 2) {
          // If our device==dest_SM_arch, choose this, or else pass
          if (deviceProp.major == best_SM_arch) {
            max_compute_perf = compute_perf;
            max_perf_device = current_device;
          }
        } else {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      }
    }

    ++current_device;
  }

  return max_perf_device;
}

// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv) {
  hipDeviceProp_t deviceProp;
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(hipSetDevice(devID));
    checkCudaErrors(hipGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  return devID;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version) {
  hipDeviceProp_t deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  int dev;

  checkCudaErrors(hipGetDevice(&dev));
  checkCudaErrors(hipGetDeviceProperties(&deviceProp, dev));

  if ((deviceProp.major > major_version) ||
      (deviceProp.major == major_version &&
       deviceProp.minor >= minor_version)) {
    printf("  GPU Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
           deviceProp.name, deviceProp.major, deviceProp.minor);
    return true;
  } else {
    printf("  No GPU device was found that can support CUDA compute capability "
           "%d.%d.\n",
           major_version, minor_version);
    return false;
  }
}
#endif

// end of CUDA Helper Functions

#endif
