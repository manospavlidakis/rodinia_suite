#ifndef KERNEL_LOAD_H_
#define KERNEL_LOAD_H_
// Load an app's GPU kernels before the timed region and measure it separately.
//
// CUDA loads kernel modules lazily, on the first launch (the default since CUDA
// 12.2; ZLUDA always does), and HIP defers code-object loading the same way. Without
// this, that one-time load is counted inside "Computation". Querying a kernel's
// attributes forces the load without launching anything, so apps call:
//
//   auto kl = kernel_load_start();
//   preload_kernels(kernelA, kernelB);   // plus helpers for kernels in other files
//   kernel_load_print(kl);               // prints "Kernel load time: X ms"
#include <chrono>
#include <cstdio>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

template <typename... K> inline void preload_kernels(K... kernels) {
#ifdef __HIPCC__
  hipFuncAttributes fa;
  int done[] = {0, ((void)hipFuncGetAttributes(
                        &fa, reinterpret_cast<const void *>(kernels)),
                    0)...};
  (void)hipGetLastError();
#else
  cudaFuncAttributes fa;
  int done[] = {0, ((void)cudaFuncGetAttributes(
                        &fa, reinterpret_cast<const void *>(kernels)),
                    0)...};
  (void)cudaGetLastError(); // the query may be unsupported (ZLUDA); the load
                            // has happened by then, so drop the error
#endif
  (void)done;
}

inline std::chrono::steady_clock::time_point kernel_load_start() {
  return std::chrono::steady_clock::now();
}

inline void kernel_load_print(std::chrono::steady_clock::time_point start) {
  std::chrono::duration<double, std::milli> ms =
      std::chrono::steady_clock::now() - start;
  fprintf(stderr, "Kernel load time: %f ms\n", ms.count());
}

#endif // KERNEL_LOAD_H_
