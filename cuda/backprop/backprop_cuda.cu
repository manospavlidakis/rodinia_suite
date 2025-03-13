

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
  cudaMalloc((void **)&warm, sizeof(double) * 100000);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaFree(warm);
  end_warmup = std::chrono::high_resolution_clock::now();
#endif

  s_compute = std::chrono::high_resolution_clock::now();
  cudaMalloc((void **)&input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void **)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float));

  // printf("Performing GPU computation\n");

  // printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum, in, hid);

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("bpnn kernel error(%d):  %s\n", error, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(partial_sum, hidden_partial_sum,
             num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);

  cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_prev_weights_cuda,
             (in + 1) * (hid + 1) * sizeof(float));

  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda, hid,
                                              input_cuda, in, input_hidden_cuda,
                                              input_prev_weights_cuda);

  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
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
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
}
