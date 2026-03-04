

// includes, system
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes, kernels
#include "backprop.h"
#include "backprop_cuda_kernel.cu"
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
std::chrono::high_resolution_clock::time_point s_b8;
std::chrono::high_resolution_clock::time_point e_b8;
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
  num_blocks = (in + 1023) / 1024;
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
  cudaMalloc((void **)&warm, sizeof(double) * 100000);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaFree(warm);
  end_warmup = std::chrono::high_resolution_clock::now();
#endif

  s_compute = std::chrono::high_resolution_clock::now();

#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif

  cudaMalloc((void **)&input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void **)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif

  bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda, input_hidden_cuda,
                                            hidden_partial_sum, in, hid);
  cudaDeviceSynchronize();

#ifdef BREAKDOWNS
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

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
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef BREAKDOWNS
  s_b4 = std::chrono::high_resolution_clock::now();
#endif

  cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b4 = std::chrono::high_resolution_clock::now();
  s_b5 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

#ifdef BREAKDOWNS
  e_b5 = std::chrono::high_resolution_clock::now();
  s_b6 = std::chrono::high_resolution_clock::now();
#endif

  bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda, hid,
                                              input_cuda, in, input_hidden_cuda,
                                              input_prev_weights_cuda);
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b6 = std::chrono::high_resolution_clock::now();
  s_b7 = std::chrono::high_resolution_clock::now();
#endif

  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef BREAKDOWNS
  e_b7 = std::chrono::high_resolution_clock::now();
  s_b8 = std::chrono::high_resolution_clock::now();
#endif

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
#ifdef BREAKDOWNS
  cudaDeviceSynchronize();
  e_b8 = std::chrono::high_resolution_clock::now();
#endif
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
#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup = end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms" << std::endl;
  cudaStreamDestroy(stream);
#endif

  std::chrono::duration<double, std::milli> compute_milli = e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

#ifdef BREAKDOWNS
  std::cerr << "##### Breakdown Computation #####" << std::endl;
  std::chrono::duration<double, std::milli> allocation1 = e_b0 - s_b0;
  std::chrono::duration<double, std::milli> allocation2 = e_b4 - s_b4;
  std::cerr << "Allocation time: " << allocation1.count() + allocation2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer1 = e_b2 - s_b2;
  std::chrono::duration<double, std::milli> transfer2 = e_b5 - s_b5;
  std::cerr << "H2D transfer time: " << transfer1.count() + transfer2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> compute1 = e_b1 - s_b1;
  std::chrono::duration<double, std::milli> compute2 = e_b6 - s_b6;
  std::cerr << "Compute time: " << compute1.count() + compute2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transferback1 = e_b3 - s_b3;
  std::chrono::duration<double, std::milli> transferback2 = e_b7 - s_b7;
  std::cerr << "D2H transfer time: " << transferback1.count() + transferback2.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> freedata = e_b8 - s_b8;
  std::cerr << "Free time: " << freedata.count() << " ms" << std::endl;
  std::cerr << "#################################" << std::endl;
#endif
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
}
