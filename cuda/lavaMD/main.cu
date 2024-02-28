#include "./main.h"
#include "cuda.h"
#include "kernel/kernel_gpu_cuda_wrapper.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;
#define WARMUP
int isInteger(char *str) {

  if (*str == '\0') {
    return 0;
  }

  for (; *str != '\0'; str++) {
    if (*str < 48 ||
        *str >
            57) { // digit characters (need to include . if checking for float)
      return 0;
    }
  }

  return 1;
}
int main(int argc, char *argv[]) {

  auto start = std::chrono::high_resolution_clock::now();
  auto start_0 = std::chrono::high_resolution_clock::now();
  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str *box_cpu;
  FOUR_VECTOR *rv_cpu;
  fp *qv_cpu;
  FOUR_VECTOR *fv_cpu;
  int nh;

  // assing default values
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  for (dim_cpu.cur_arg = 1; dim_cpu.cur_arg < argc; dim_cpu.cur_arg++) {
    // check if -boxes1d
    if (strcmp(argv[dim_cpu.cur_arg], "-boxes1d") == 0) {
      // check if value provided
      if (argc >= dim_cpu.cur_arg + 1) {
        // check if value is a number
        if (isInteger(argv[dim_cpu.cur_arg + 1]) == 1) {
          dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg + 1]);
          if (dim_cpu.boxes1d_arg < 0) {
            printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
            return 0;
          }
          dim_cpu.cur_arg = dim_cpu.cur_arg + 1;
        }
        // value is not a number
        else {
          printf("ERROR: Value to -boxes1d parameter in not a number\n");
          return 0;
        }
      }
      // value not provided
      else {
        printf("ERROR: Missing value to -boxes1d parameter\n");
        return 0;
      }
    }
    // unknown
    else {
      printf("ERROR: Unknown parameter\n");
      return 0;
    }
  }

  par_cpu.alpha = 0.5;

  // total number of boxes
  dim_cpu.number_boxes =
      dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  // allocate boxes
  box_cpu = (box_str *)malloc(dim_cpu.box_mem);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for (i = 0; i < dim_cpu.boxes1d_arg; i++) {
    // home boxes in y direction
    for (j = 0; j < dim_cpu.boxes1d_arg; j++) {
      // home boxes in x direction
      for (k = 0; k < dim_cpu.boxes1d_arg; k++) {

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for (l = -1; l < 2; l++) {
          // neighbor boxes in y direction
          for (m = -1; m < 2; m++) {
            // neighbor boxes in x direction
            for (n = -1; n < 2; n++) {

              // check if (this neighbor exists) and (it is not the same as home
              // box)
              if ((((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0) == true &&
                   ((i + l) < dim_cpu.boxes1d_arg &&
                    (j + m) < dim_cpu.boxes1d_arg &&
                    (k + n) < dim_cpu.boxes1d_arg) == true) &&
                  (l == 0 && m == 0 && n == 0) == false) {
                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
                box_cpu[nh].nei[box_cpu[nh].nn].number =
                    (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg *
                     dim_cpu.boxes1d_arg) +
                    (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                    box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset =
                    box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;
              }

            } // neighbor boxes in x direction
          }   // neighbor boxes in y direction
        }     // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    }   // home boxes in y direction
  }     // home boxes in z direction

  // random generator seed set to random value - time in this case
  srand(100);

  // input (distances)
  rv_cpu = (FOUR_VECTOR *)malloc(dim_cpu.space_mem);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    rv_cpu[i].v = (rand() % 10 + 1) / 10.0;
    rv_cpu[i].x = (rand() % 10 + 1) / 10.0;
    rv_cpu[i].y = (rand() % 10 + 1) / 10.0;
    rv_cpu[i].z = (rand() % 10 + 1) / 10.0;
  }

  // input (charge)
  qv_cpu = (fp *)malloc(dim_cpu.space_mem2);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    qv_cpu[i] = (rand() % 10 + 1) / 10.0;
  }

  // output (forces)
  fv_cpu = (FOUR_VECTOR *)malloc(dim_cpu.space_mem);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fv_cpu[i].v = 0;
    fv_cpu[i].x = 0;
    fv_cpu[i].y = 0;
    fv_cpu[i].z = 0;
  }
  auto end_0 = std::chrono::high_resolution_clock::now();
#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  char *warm;
  cudaMalloc((void **)&warm, sizeof(char));
  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  s_compute = std::chrono::high_resolution_clock::now();

  kernel_gpu_cuda_wrapper(par_cpu, dim_cpu, box_cpu, rv_cpu, qv_cpu, fv_cpu);

  // dump results
#ifdef OUTPUT
  FILE *fptr;
  fptr = fopen("nat_result.txt", "w");
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y,
            fv_cpu[i].z);
  }
  fclose(fptr);
#endif
  e_compute = std::chrono::high_resolution_clock::now();

  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli_0 = end_0 - start_0;
  std::cerr << "Init time: " << elapsed_milli_0.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

  std::chrono::duration<double, std::milli> elapsed_milli = end - start;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;

#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
  // free warmup
  cudaFree(warm);
#endif
  return 0.0;
}
