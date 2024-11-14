/***********************************************************************************

 Implementing Breadth first search on CUDA using algorithm given in HiPC'07

 paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright
 (c) 2008 International Institute of Information Technology - Hyderabad. All
 rights reserved.

  Permission to use, copy, modify and distribute this
 software and its documentation for educational purpose is hereby granted
 without fee, provided that the above copyright notice and this permission
 notice appear in all copies of this software and that you do not sell the
 software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY
 KIND,EXPRESS, IMPLIED OR OTHERWISE.

  Created by Pawan Harish.

 ************************************************************************************/
#include "../../common/util.h"
#include "hip/hip_runtime.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS_PER_BLOCK 512
#define WARMUP
//#define DEBUG
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;

int no_of_nodes;
int edge_list_size;
FILE *fp;
char *warm;

// Structure to hold a node information
struct Node {
  int starting;
  int no_of_edges;
};


#define BREAKDOWNS
  
#ifdef BREAKDOWNS
std::chrono::high_resolution_clock::time_point s_b0;
std::chrono::high_resolution_clock::time_point e_b0;
std::chrono::high_resolution_clock::time_point s_b1;
std::chrono::high_resolution_clock::time_point e_b1;
std::chrono::high_resolution_clock::time_point s_b2;
std::chrono::high_resolution_clock::time_point e_b2;
std::chrono::high_resolution_clock::time_point s_b3;
std::chrono::high_resolution_clock::time_point e_b3;
      
#endif
#include "kernel.hip.cu"
#include "kernel2.hip.cu"

void BFSGraph(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  no_of_nodes = 0;
  edge_list_size = 0;
  BFSGraph(argc, argv);
}

void Usage(int argc, char **argv) {

  fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}
////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
  auto start_all = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  char *input_f;
  if (argc != 2) {
    Usage(argc, argv);
    exit(0);
  }

  input_f = argv[1];
  // Read in Graph from a file
  fp = fopen(input_f, "r");
  if (!fp) {
    printf("Error Reading graph file\n");
    return;
  }

  int source = 0;

  fscanf(fp, "%d", &no_of_nodes);

  int num_of_blocks = 1;
  int num_of_threads_per_block = no_of_nodes;

  // Make execution Parameters according to the number of nodes
  // Distribute threads across multiple Blocks if necessary
  if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
    num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
    num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
  }

  // allocate host memory
  Node *h_graph_nodes = (Node *)alignedMalloc(sizeof(Node) * no_of_nodes);
  bool *h_graph_mask = (bool *)alignedMalloc(sizeof(bool) * no_of_nodes);
  bool *h_updating_graph_mask =
      (bool *)alignedMalloc(sizeof(bool) * no_of_nodes);
  bool *h_graph_visited = (bool *)alignedMalloc(sizeof(bool) * no_of_nodes);

  int start, edgeno;
  // initalize the memory
  for (unsigned int i = 0; i < no_of_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i] = false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i] = false;
  }

  // read the source node from the file
  fscanf(fp, "%d", &source);
  source = 0;

  // set the source node as true in the mask
  h_graph_mask[source] = true;
  h_graph_visited[source] = true;

  fscanf(fp, "%d", &edge_list_size);

  int id, cost;
  int *h_graph_edges = (int *)alignedMalloc(sizeof(int) * edge_list_size);
  for (int i = 0; i < edge_list_size; i++) {
    fscanf(fp, "%d", &id);
    fscanf(fp, "%d", &cost);
    h_graph_edges[i] = id;
  }

  if (fp)
    fclose(fp);

  // allocate mem for the result on host side
  int *h_cost = (int *)alignedMalloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++)
    h_cost[i] = -1;
  h_cost[source] = 0;
  auto end_0 = std::chrono::high_resolution_clock::now();
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  hipMalloc((void **)&warm, sizeof(char));
  hipStream_t stream;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  s_compute = std::chrono::high_resolution_clock::now();
#ifdef BREAKDOWNS
  s_b0 = std::chrono::high_resolution_clock::now();
#endif
  // Copy the Node list to device memory
  Node *d_graph_nodes;
  hipMalloc((void **)&d_graph_nodes, sizeof(Node) * no_of_nodes);

  // Copy the Edge List to device Memory
  int *d_graph_edges;
  hipMalloc((void **)&d_graph_edges, sizeof(int) * edge_list_size);

  // Copy the Mask to device memory
  bool *d_graph_mask;
  hipMalloc((void **)&d_graph_mask, sizeof(bool) * no_of_nodes);

  bool *d_updating_graph_mask;
  hipMalloc((void **)&d_updating_graph_mask, sizeof(bool) * no_of_nodes);

  // Copy the Visited nodes array to device memory
  bool *d_graph_visited;
  hipMalloc((void **)&d_graph_visited, sizeof(bool) * no_of_nodes);

  // allocate device memory for result
  int *d_cost;
  hipMalloc((void **)&d_cost, sizeof(int) * no_of_nodes);

  // make a bool to check if the execution is over
  bool *d_over;

  hipMalloc((void **)&d_over, sizeof(bool));

#ifdef BREAKDOWNS
  hipDeviceSynchronize();
  e_b0 = std::chrono::high_resolution_clock::now();
  s_b2 = std::chrono::high_resolution_clock::now();
#endif
  // nodelist
  hipMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes,
            hipMemcpyHostToDevice);
  // edgelist
  hipMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size,
            hipMemcpyHostToDevice);
  // mask
  hipMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes,
            hipMemcpyHostToDevice);
  hipMemcpy(d_updating_graph_mask, h_updating_graph_mask,
            sizeof(bool) * no_of_nodes, hipMemcpyHostToDevice);
  // visited nodes
  hipMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes,
            hipMemcpyHostToDevice);
  // device memory for result
  hipMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, hipMemcpyHostToDevice);
#ifdef BREAKDOWNS
  e_b2 = std::chrono::high_resolution_clock::now();
  s_b1 = std::chrono::high_resolution_clock::now();
#endif
  // setup execution parameters
  dim3 grid(num_of_blocks, 1, 1);
  dim3 threads(num_of_threads_per_block, 1, 1);

  int k = 0;
  bool stop;

  do {
    // if no thread changes this value then the loop stops
    stop = false;
    hipMemcpy(d_over, &stop, sizeof(bool), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(Kernel, dim3(grid), dim3(threads), 0, 0, d_graph_nodes,
                       d_graph_edges, d_graph_mask, d_updating_graph_mask,
                       d_graph_visited, d_cost, no_of_nodes);
    // check if kernel execution generated and error

    hipLaunchKernelGGL(Kernel2, dim3(grid), dim3(threads), 0, 0, d_graph_mask,
                       d_updating_graph_mask, d_graph_visited, d_over,
                       no_of_nodes);
    // check if kernel execution generated and error

    hipMemcpy(&stop, d_over, sizeof(bool), hipMemcpyDeviceToHost);
    k++;
  } while (stop);
#ifdef BREAKDOWNS
  hipDeviceSynchronize();
  e_b1 = std::chrono::high_resolution_clock::now();
  s_b3 = std::chrono::high_resolution_clock::now();
#endif
  // copy result from device to host
  hipMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, hipMemcpyDeviceToHost);
#ifdef BREAKDOWNS
  e_b3 = std::chrono::high_resolution_clock::now();
#endif
#ifdef OUTPUT
  // Store the result into a file
  FILE *fpo = fopen("result.txt", "w");
  for (int i = 0; i < no_of_nodes; i++)
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  fclose(fpo);
  printf("Result stored in result.txt\n");
#endif
  hipFree(d_graph_nodes);
  hipFree(d_graph_edges);
  hipFree(d_graph_mask);
  hipFree(d_updating_graph_mask);
  hipFree(d_graph_visited);
  hipFree(d_cost);
  e_compute = std::chrono::high_resolution_clock::now();

  // cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);

  auto end_all = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli_0 = end_0 - start_0;
  std::cerr << "Init time: " << elapsed_milli_0.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> elapsed_milli = end_all - start_all;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
  // free warmup
  hipFree(warm);
#endif
#ifdef BREAKDOWNS
  std::cerr << " ##### Breakdown Computation #####" << std::endl;
  std::chrono::duration<double, std::milli> allocation = e_b0 - s_b0;
  std::cerr << "Allocation time: " << allocation.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer = e_b2 - s_b2;
  std::cerr << "Transfer time: " << transfer.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> compute = e_b1 - s_b1;
  std::cerr << "Compute time: " << compute.count() << " ms" << std::endl;
  std::chrono::duration<double, std::milli> transfer2 = e_b3 - s_b3;
  std::cerr << "Transfer Back time: " << transfer2.count() << " ms"
            << std::endl;
  std::cerr << " #################################" << std::endl;
#endif
}
