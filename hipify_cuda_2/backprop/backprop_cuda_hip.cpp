

// includes, system
#include <chrono>
#include <hip/hip_runtime.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/helper_hip.h"
// includes, kernels
#include "backprop.h"
#include "backprop_cuda_kernel_hip.cpp"
#include "imagenet.h"

#ifdef BREAKDOWNS
std::chrono::high_resolution_clock::time_point s_b0;
std::chrono::high_resolution_clock::time_point e_b0;
std::chrono::high_resolution_clock::time_point s_b1;
std::chrono::high_resolution_clock::time_point e_b1;
std::chrono::high_resolution_clock::time_point s_b2;
std::chrono::high_resolution_clock::time_point e_b2;
std::chrono::high_resolution_clock::time_point s_b3;
std::chrono::high_resolution_clock::time_point e_b3;
std::chrono::high_resolution_clock::time_point s_b4;
std::chrono::high_resolution_clock::time_point e_b4;
std::chrono::high_resolution_clock::time_point s_b5;
std::chrono::high_resolution_clock::time_point e_b5;
std::chrono::high_resolution_clock::time_point s_b6;
std::chrono::high_resolution_clock::time_point e_b6;
std::chrono::high_resolution_clock::time_point s_b7;
std::chrono::high_resolution_clock::time_point e_b7;
#endif

#define WARMUP
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  auto start_all = std::chrono::high_resolution_clock::now();
  setup(argc, argv);
  auto end_all = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli = end_all - start_all;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
}

extern "C" void bpnn_train_cuda(BPNN *net, float *eo, float *eh) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 1024;
  dim3 grid(1, num_blocks);
  dim3 threads(16, 16);

  input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim =
      (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy
  // using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  double *warm;
  HIP_CHECK(hipMalloc((void **)&warm, sizeof(double) * 100000));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  HIP_CHECK(hipFree(warm));
  end_warmup = std::chrono::high_resolution_clock::now();
#endif

  s_compute = std::chrono::high_resolution_clock::now();

#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMalloc((void **)&input_cuda, (in + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void **)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float)));

#ifdef BREAKDOWNS
  HIP_CHECK(hipDeviceSynchronize());
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), hipMemcpyHostToDevice));

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda, input_hidden_cuda,
                                            hidden_partial_sum, in, hid);
  HIP_CHECK(hipDeviceSynchronize());

#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), hipMemcpyDeviceToHost));

#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);

#ifdef BREAKDOWNS
  s_b4 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void **)&input_prev_weights_cuda,
             (in + 1) * (hid + 1) * sizeof(float)));

#ifdef BREAKDOWNS
  e_b4 = std::chrono::high_resolution_clock::now();
  s_b5 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
             hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), hipMemcpyHostToDevice));

#ifdef BREAKDOWNS
  e_b5 = std::chrono::high_resolution_clock::now();
  s_b6 = std::chrono::high_resolution_clock::now();
#endif

  bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda, hid,
                                              input_cuda, in, input_hidden_cuda,
                                              input_prev_weights_cuda);
#ifdef BREAKDOWNS
  HIP_CHECK(hipDeviceSynchronize());
  e_b6 = std::chrono::high_resolution_clock::now();
  s_b7 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
             hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(input_weights_one_dim, input_hidden_cuda,
             (in + 1) * (hid + 1) * sizeof(float), hipMemcpyDeviceToHost));

#ifdef BREAKDOWNS
  e_b7 = std::chrono::high_resolution_clock::now();
#endif

  HIP_CHECK(hipFree(input_cuda));
  HIP_CHECK(hipFree(output_hidden_cuda));
  HIP_CHECK(hipFree(input_hidden_cuda));
  HIP_CHECK(hipFree(hidden_partial_sum));
  HIP_CHECK(hipFree(input_prev_weights_cuda));
  HIP_CHECK(hipFree(hidden_delta_cuda));
  e_compute = std::chrono::high_resolution_clock::now();
  // Open a file for output
  std::ofstream outfile("result.txt");
  if (!outfile) {
    std::cerr << "Failed to open file for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < 1000; i++) {
    outfile << input_weights_one_dim[i] << std::endl;
  }

  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
#endif
#ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown kernel wrapper 1 #####" << std::endl;
  std::chrono::duration<double, std::milli> allocation1 = e_b0 - s_b0;
  std::chrono::duration<double, std::milli> allocation2 = e_b4 - s_b4;
  std::cerr << "Allocation time: " << allocation1.count() + allocation2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer1 = e_b2 - s_b2;
  std::chrono::duration<double, std::milli> transfer2 = e_b5 - s_b5;
  std::cerr << "Transfer time: " << transfer1.count() + transfer2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> compute1 = e_b1 - s_b1;
  std::chrono::duration<double, std::milli> compute2 = e_b6 - s_b6;
  std::cerr << "Compute time: " << compute1.count() + compute2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transferback1 = e_b3 - s_b3;
  std::chrono::duration<double, std::milli> transferback2 = e_b7 - s_b7;
  std::cerr << "Transfer Back time: " << transferback1.count() + transferback2.count() << " ms"
            << std::endl;
  std::cerr << " #################################" << std::endl;
#endif
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
}
