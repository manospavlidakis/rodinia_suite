#include "define.c"

#include <AVI/avilib.h>
#include <AVI/avimod.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
params_common_change common_change;
__constant__ params_common_change d_common_change;

params_common common;
__constant__ params_common d_common;

params_unique unique[ALL_POINTS]; // cannot determine size dynamically so choose
                                  // more than usually needed
__constant__ params_unique d_unique[ALL_POINTS];

__global__ void kernel() {

  fp *d_in;
  int rot_row;
  int rot_col;
  int in2_rowlow;
  int in2_collow;
  int ic;
  int jc;
  int jp1;
  int ja1, ja2;
  int ip1;
  int ia1, ia2;
  int ja, jb;
  int ia, ib;
  float s;
  int i;
  int j;
  int row;
  int col;
  int ori_row;
  int ori_col;
  int position;
  float sum;
  int pos_ori;
  float temp;
  float temp2;
  int location;
  int cent;
  int tMask_row;
  int tMask_col;
  float largest_value_current = 0;
  float largest_value = 0;
  int largest_coordinate_current = 0;
  int largest_coordinate = 0;
  float fin_max_val = 0;
  int fin_max_coo = 0;
  int largest_row;
  int largest_col;
  int offset_row;
  int offset_col;
  __shared__ float in_partial_sum[51];     // WATCH THIS !!! HARDCODED VALUE
  __shared__ float in_sqr_partial_sum[51]; // WATCH THIS !!! HARDCODED VALUE
  __shared__ float in_final_sum;
  __shared__ float in_sqr_final_sum;
  float mean;
  float mean_sqr;
  float variance;
  float deviation;
  __shared__ float denomT;
  __shared__ float par_max_val[131]; // WATCH THIS !!! HARDCODED VALUE
  __shared__ int par_max_coo[131];   // WATCH THIS !!! HARDCODED VALUE
  int pointer;
  __shared__ float d_in_mod_temp[2601];
  int ori_pointer;
  int loc_pointer;

  int bx = blockIdx.x;  // get current horizontal block index (0-n)
  int tx = threadIdx.x; // get current horizontal thread index (0-n)
  int ei_new;

  // generate templates based on the first frame only
  if (d_common_change.frame_no == 0) {

    // pointers to: current template for current point
    d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

    // uptade temporary endo/epi row/col coordinates (in each block
    // corresponding to point, narrow work to one thread)
    ei_new = tx;
    if (ei_new == 0) {

      // update temporary row/col coordinates
      pointer =
          d_unique[bx].point_no * d_common.no_frames + d_common_change.frame_no;
      d_unique[bx].d_tRowLoc[pointer] =
          d_unique[bx].d_Row[d_unique[bx].point_no];
      d_unique[bx].d_tColLoc[pointer] =
          d_unique[bx].d_Col[d_unique[bx].point_no];
    }

    // work
    ei_new = tx;
    while (ei_new < d_common.in_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in_rows == 0) {
        row = d_common.in_rows - 1;
        col = col - 1;
      }

      // figure out row/col location in corresponding new template area in image
      // and give to every thread (get top left corner and progress down and
      // right)
      ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
      ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
      ori_pointer = ori_col * d_common.frame_rows + ori_row;

      // update template
      d_in[col * d_common.in_rows + row] = d_common_change.d_frame[ori_pointer];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }
  }

  // process points in all frames except for the first one
  if (d_common_change.frame_no != 0) {

    in2_rowlow = d_unique[bx].d_Row[d_unique[bx].point_no] -
                 d_common.sSize; // (1 to n+1)
    in2_collow = d_unique[bx].d_Col[d_unique[bx].point_no] - d_common.sSize;

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_rows == 0) {
        row = d_common.in2_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + in2_rowlow - 1;
      ori_col = col + in2_collow - 1;
      d_unique[bx].d_in2[ei_new] =
          d_common_change.d_frame[ori_col * d_common.frame_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // variables
    d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

    // work
    ei_new = tx;
    while (ei_new < d_common.in_elem) {

      // figure out row/col location in padded array
      row = (ei_new + 1) % d_common.in_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in_rows == 0) {
        row = d_common.in_rows - 1;
        col = col - 1;
      }

      // execution
      rot_row = (d_common.in_rows - 1) - row;
      rot_col = (d_common.in_rows - 1) - col;
      d_in_mod_temp[ei_new] = d_in[rot_col * d_common.in_rows + rot_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.conv_elem) {

      // figure out row/col location in array
      ic = (ei_new + 1) % d_common.conv_rows;     // (1-n)
      jc = (ei_new + 1) / d_common.conv_rows + 1; // (1-n)
      if ((ei_new + 1) % d_common.conv_rows == 0) {
        ic = d_common.conv_rows;
        jc = jc - 1;
      }

      //
      j = jc + d_common.joffset;
      jp1 = j + 1;
      if (d_common.in2_cols < jp1) {
        ja1 = jp1 - d_common.in2_cols;
      } else {
        ja1 = 1;
      }
      if (d_common.in_cols < j) {
        ja2 = d_common.in_cols;
      } else {
        ja2 = j;
      }

      i = ic + d_common.ioffset;
      ip1 = i + 1;

      if (d_common.in2_rows < ip1) {
        ia1 = ip1 - d_common.in2_rows;
      } else {
        ia1 = 1;
      }
      if (d_common.in_rows < i) {
        ia2 = d_common.in_rows;
      } else {
        ia2 = i;
      }

      s = 0;

      for (ja = ja1; ja <= ja2; ja++) {
        jb = jp1 - ja;
        for (ia = ia1; ia <= ia2; ia++) {
          ib = ip1 - ia;
          s = s + d_in_mod_temp[d_common.in_rows * (ja - 1) + ia - 1] *
                      d_unique[bx].d_in2[d_common.in2_rows * (jb - 1) + ib - 1];
        }
      }

      // d_unique[bx].d_conv[d_common.conv_rows*(jc-1)+ic-1] = s;
      d_unique[bx].d_conv[ei_new] = s;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_pad_cumv_elem) {

      // figure out row/col location in padded array
      row = (ei_new + 1) % d_common.in2_pad_cumv_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_pad_cumv_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_pad_cumv_rows == 0) {
        row = d_common.in2_pad_cumv_rows - 1;
        col = col - 1;
      }

      // execution
      if (row > (d_common.in2_pad_add_rows -
                 1) && // do if has numbers in original array
          row < (d_common.in2_pad_add_rows + d_common.in2_rows) &&
          col > (d_common.in2_pad_add_cols - 1) &&
          col < (d_common.in2_pad_add_cols + d_common.in2_cols)) {
        ori_row = row - d_common.in2_pad_add_rows;
        ori_col = col - d_common.in2_pad_add_cols;
        d_unique[bx].d_in2_pad_cumv[ei_new] =
            d_unique[bx].d_in2[ori_col * d_common.in2_rows + ori_row];
      } else { // do if otherwise
        d_unique[bx].d_in2_pad_cumv[ei_new] = 0;
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_pad_cumv_cols) {

      // figure out column position
      pos_ori = ei_new * d_common.in2_pad_cumv_rows;

      // variables
      sum = 0;

      // loop through all rows
      for (position = pos_ori; position < pos_ori + d_common.in2_pad_cumv_rows;
           position = position + 1) {
        d_unique[bx].d_in2_pad_cumv[position] =
            d_unique[bx].d_in2_pad_cumv[position] + sum;
        sum = d_unique[bx].d_in2_pad_cumv[position];
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_pad_cumv_sel_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_pad_cumv_sel_rows - 1; // (0-n) row
      col =
          (ei_new + 1) / d_common.in2_pad_cumv_sel_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_pad_cumv_sel_rows == 0) {
        row = d_common.in2_pad_cumv_sel_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
      ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
      d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
          d_unique[bx]
              .d_in2_pad_cumv[ori_col * d_common.in2_pad_cumv_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_sub_cumh_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_sub_cumh_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_sub_cumh_rows == 0) {
        row = d_common.in2_sub_cumh_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
      ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
      d_unique[bx].d_in2_sub_cumh[ei_new] =
          d_unique[bx]
              .d_in2_pad_cumv[ori_col * d_common.in2_pad_cumv_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_elem) {

      // subtract
      d_unique[bx].d_in2_sub_cumh[ei_new] =
          d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
          d_unique[bx].d_in2_sub_cumh[ei_new];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_rows) {

      // figure out row position
      pos_ori = ei_new;

      // variables
      sum = 0;

      // loop through all rows
      for (position = pos_ori; position < pos_ori + d_common.in2_sub_cumh_elem;
           position = position + d_common.in2_sub_cumh_rows) {
        d_unique[bx].d_in2_sub_cumh[position] =
            d_unique[bx].d_in2_sub_cumh[position] + sum;
        sum = d_unique[bx].d_in2_sub_cumh[position];
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_sel_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_sub_cumh_sel_rows - 1; // (0-n) row
      col =
          (ei_new + 1) / d_common.in2_sub_cumh_sel_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_sub_cumh_sel_rows == 0) {
        row = d_common.in2_sub_cumh_sel_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
      ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
      d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
          d_unique[bx]
              .d_in2_sub_cumh[ori_col * d_common.in2_sub_cumh_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_sub2_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_sub2_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_sub2_rows == 0) {
        row = d_common.in2_sub2_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
      ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
      d_unique[bx].d_in2_sub2[ei_new] =
          d_unique[bx]
              .d_in2_sub_cumh[ori_col * d_common.in2_sub_cumh_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();
    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      // subtract
      d_unique[bx].d_in2_sub2[ei_new] =
          d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
          d_unique[bx].d_in2_sub2[ei_new];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sqr_elem) {

      temp = d_unique[bx].d_in2[ei_new];
      d_unique[bx].d_in2_sqr[ei_new] = temp * temp;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_pad_cumv_elem) {

      // figure out row/col location in padded array
      row = (ei_new + 1) % d_common.in2_pad_cumv_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_pad_cumv_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_pad_cumv_rows == 0) {
        row = d_common.in2_pad_cumv_rows - 1;
        col = col - 1;
      }

      // execution
      if (row > (d_common.in2_pad_add_rows -
                 1) && // do if has numbers in original array
          row < (d_common.in2_pad_add_rows + d_common.in2_sqr_rows) &&
          col > (d_common.in2_pad_add_cols - 1) &&
          col < (d_common.in2_pad_add_cols + d_common.in2_sqr_cols)) {
        ori_row = row - d_common.in2_pad_add_rows;
        ori_col = col - d_common.in2_pad_add_cols;
        d_unique[bx].d_in2_pad_cumv[ei_new] =
            d_unique[bx].d_in2_sqr[ori_col * d_common.in2_sqr_rows + ori_row];
      } else { // do if otherwise
        d_unique[bx].d_in2_pad_cumv[ei_new] = 0;
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_pad_cumv_cols) {

      // figure out column position
      pos_ori = ei_new * d_common.in2_pad_cumv_rows;

      // variables
      sum = 0;

      // loop through all rows
      for (position = pos_ori; position < pos_ori + d_common.in2_pad_cumv_rows;
           position = position + 1) {
        d_unique[bx].d_in2_pad_cumv[position] =
            d_unique[bx].d_in2_pad_cumv[position] + sum;
        sum = d_unique[bx].d_in2_pad_cumv[position];
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_pad_cumv_sel_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_pad_cumv_sel_rows - 1; // (0-n) row
      col =
          (ei_new + 1) / d_common.in2_pad_cumv_sel_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_pad_cumv_sel_rows == 0) {
        row = d_common.in2_pad_cumv_sel_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
      ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
      d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
          d_unique[bx]
              .d_in2_pad_cumv[ori_col * d_common.in2_pad_cumv_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_sub_cumh_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_sub_cumh_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_sub_cumh_rows == 0) {
        row = d_common.in2_sub_cumh_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
      ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
      d_unique[bx].d_in2_sub_cumh[ei_new] =
          d_unique[bx]
              .d_in2_pad_cumv[ori_col * d_common.in2_pad_cumv_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_elem) {

      // subtract
      d_unique[bx].d_in2_sub_cumh[ei_new] =
          d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
          d_unique[bx].d_in2_sub_cumh[ei_new];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_rows) {

      // figure out row position
      pos_ori = ei_new;

      // variables
      sum = 0;

      // loop through all rows
      for (position = pos_ori; position < pos_ori + d_common.in2_sub_cumh_elem;
           position = position + d_common.in2_sub_cumh_rows) {
        d_unique[bx].d_in2_sub_cumh[position] =
            d_unique[bx].d_in2_sub_cumh[position] + sum;
        sum = d_unique[bx].d_in2_sub_cumh[position];
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub_cumh_sel_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_sub_cumh_sel_rows - 1; // (0-n) row
      col =
          (ei_new + 1) / d_common.in2_sub_cumh_sel_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_sub_cumh_sel_rows == 0) {
        row = d_common.in2_sub_cumh_sel_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
      ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
      d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
          d_unique[bx]
              .d_in2_sub_cumh[ori_col * d_common.in2_sub_cumh_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in2_sub2_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in2_sub2_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in2_sub2_rows == 0) {
        row = d_common.in2_sub2_rows - 1;
        col = col - 1;
      }

      // figure out corresponding location in old matrix and copy values to new
      // matrix
      ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
      ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
      d_unique[bx].d_in2_sqr_sub2[ei_new] =
          d_unique[bx]
              .d_in2_sub_cumh[ori_col * d_common.in2_sub_cumh_rows + ori_row];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      // subtract
      d_unique[bx].d_in2_sqr_sub2[ei_new] =
          d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
          d_unique[bx].d_in2_sqr_sub2[ei_new];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      temp = d_unique[bx].d_in2_sub2[ei_new];
      temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] -
              (temp * temp / d_common.in_elem);
      if (temp2 < 0) {
        temp2 = 0;
      }
      d_unique[bx].d_in2_sqr_sub2[ei_new] = sqrt(temp2);

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in_sqr_elem) {

      temp = d_in[ei_new];
      d_unique[bx].d_in_sqr[ei_new] = temp * temp;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }
    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in_cols) {

      sum = 0;
      for (i = 0; i < d_common.in_rows; i++) {

        sum = sum + d_in[ei_new * d_common.in_rows + i];
      }
      in_partial_sum[ei_new] = sum;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    ei_new = tx;
    while (ei_new < d_common.in_sqr_rows) {

      sum = 0;
      for (i = 0; i < d_common.in_sqr_cols; i++) {

        sum = sum + d_unique[bx].d_in_sqr[ei_new + d_common.in_sqr_rows * i];
      }
      in_sqr_partial_sum[ei_new] = sum;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    if (tx == 0) {

      in_final_sum = 0;
      for (i = 0; i < d_common.in_cols; i++) {
        in_final_sum = in_final_sum + in_partial_sum[i];
      }

    } else if (tx == 1) {

      in_sqr_final_sum = 0;
      for (i = 0; i < d_common.in_sqr_cols; i++) {
        in_sqr_final_sum = in_sqr_final_sum + in_sqr_partial_sum[i];
      }
    }

    __syncthreads();

    if (tx == 0) {

      mean = in_final_sum /
             d_common.in_elem; // gets mean (average) value of element in ROI
      mean_sqr = mean * mean;
      variance = (in_sqr_final_sum / d_common.in_elem) -
                 mean_sqr;        // gets variance of ROI
      deviation = sqrt(variance); // gets standard deviation of ROI

      denomT = sqrt(float(d_common.in_elem - 1)) * deviation;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      d_unique[bx].d_in2_sqr_sub2[ei_new] =
          d_unique[bx].d_in2_sqr_sub2[ei_new] * denomT;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.conv_elem) {

      d_unique[bx].d_conv[ei_new] =
          d_unique[bx].d_conv[ei_new] -
          d_unique[bx].d_in2_sub2[ei_new] * in_final_sum / d_common.in_elem;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.in2_sub2_elem) {

      d_unique[bx].d_in2_sqr_sub2[ei_new] =
          d_unique[bx].d_conv[ei_new] / d_unique[bx].d_in2_sqr_sub2[ei_new];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    cent = d_common.sSize + d_common.tSize + 1;
    if (d_common_change.frame_no == 0) {
      tMask_row = cent + d_unique[bx].d_Row[d_unique[bx].point_no] -
                  d_unique[bx].d_Row[d_unique[bx].point_no] - 1;
      tMask_col = cent + d_unique[bx].d_Col[d_unique[bx].point_no] -
                  d_unique[bx].d_Col[d_unique[bx].point_no] - 1;
    } else {
      pointer = d_common_change.frame_no - 1 +
                d_unique[bx].point_no * d_common.no_frames;
      tMask_row = cent + d_unique[bx].d_tRowLoc[pointer] -
                  d_unique[bx].d_Row[d_unique[bx].point_no] - 1;
      tMask_col = cent + d_unique[bx].d_tColLoc[pointer] -
                  d_unique[bx].d_Col[d_unique[bx].point_no] - 1;
    }

    // work
    ei_new = tx;
    while (ei_new < d_common.tMask_elem) {

      location = tMask_col * d_common.tMask_rows + tMask_row;

      if (ei_new == location) {
        d_unique[bx].d_tMask[ei_new] = 1;
      } else {
        d_unique[bx].d_tMask[ei_new] = 0;
      }

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    // work
    ei_new = tx;
    while (ei_new < d_common.mask_conv_elem) {

      // figure out row/col location in array
      ic = (ei_new + 1) % d_common.mask_conv_rows;     // (1-n)
      jc = (ei_new + 1) / d_common.mask_conv_rows + 1; // (1-n)
      if ((ei_new + 1) % d_common.mask_conv_rows == 0) {
        ic = d_common.mask_conv_rows;
        jc = jc - 1;
      }

      //
      j = jc + d_common.mask_conv_joffset;
      jp1 = j + 1;
      if (d_common.mask_cols < jp1) {
        ja1 = jp1 - d_common.mask_cols;
      } else {
        ja1 = 1;
      }
      if (d_common.tMask_cols < j) {
        ja2 = d_common.tMask_cols;
      } else {
        ja2 = j;
      }

      i = ic + d_common.mask_conv_ioffset;
      ip1 = i + 1;

      if (d_common.mask_rows < ip1) {
        ia1 = ip1 - d_common.mask_rows;
      } else {
        ia1 = 1;
      }
      if (d_common.tMask_rows < i) {
        ia2 = d_common.tMask_rows;
      } else {
        ia2 = i;
      }

      s = 0;

      for (ja = ja1; ja <= ja2; ja++) {
        jb = jp1 - ja;
        for (ia = ia1; ia <= ia2; ia++) {
          ib = ip1 - ia;
          s = s +
              d_unique[bx].d_tMask[d_common.tMask_rows * (ja - 1) + ia - 1] * 1;
        }
      }

      // //d_unique[bx].d_mask_conv[d_common.mask_conv_rows*(jc-1)+ic-1] = s;
      d_unique[bx].d_mask_conv[ei_new] =
          d_unique[bx].d_in2_sqr_sub2[ei_new] * s;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    ei_new = tx;
    while (ei_new < d_common.mask_conv_rows) {

      for (i = 0; i < d_common.mask_conv_cols; i++) {
        largest_coordinate_current = ei_new * d_common.mask_conv_rows + i;
        largest_value_current =
            abs(d_unique[bx].d_mask_conv[largest_coordinate_current]);
        if (largest_value_current > largest_value) {
          largest_coordinate = largest_coordinate_current;
          largest_value = largest_value_current;
        }
      }
      par_max_coo[ei_new] = largest_coordinate;
      par_max_val[ei_new] = largest_value;

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }

    __syncthreads();

    if (tx == 0) {

      for (i = 0; i < d_common.mask_conv_rows; i++) {
        if (par_max_val[i] > fin_max_val) {
          fin_max_val = par_max_val[i];
          fin_max_coo = par_max_coo[i];
        }
      }

      // convert coordinate to row/col form
      largest_row =
          (fin_max_coo + 1) % d_common.mask_conv_rows - 1;       // (0-n) row
      largest_col = (fin_max_coo + 1) / d_common.mask_conv_rows; // (0-n) column
      if ((fin_max_coo + 1) % d_common.mask_conv_rows == 0) {
        largest_row = d_common.mask_conv_rows - 1;
        largest_col = largest_col - 1;
      }

      // calculate offset
      largest_row = largest_row + 1; // compensate to match MATLAB format (1-n)
      largest_col = largest_col + 1; // compensate to match MATLAB format (1-n)
      offset_row =
          largest_row - d_common.in_rows - (d_common.sSize - d_common.tSize);
      offset_col =
          largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);
      pointer =
          d_common_change.frame_no + d_unique[bx].point_no * d_common.no_frames;
      d_unique[bx].d_tRowLoc[pointer] =
          d_unique[bx].d_Row[d_unique[bx].point_no] + offset_row;
      d_unique[bx].d_tColLoc[pointer] =
          d_unique[bx].d_Col[d_unique[bx].point_no] + offset_col;
    }

    __syncthreads();
  }

  // if the last frame in the bath, update template
  if (d_common_change.frame_no != 0 && (d_common_change.frame_no) % 10 == 0) {

    // update coordinate
    loc_pointer =
        d_unique[bx].point_no * d_common.no_frames + d_common_change.frame_no;
    d_unique[bx].d_Row[d_unique[bx].point_no] =
        d_unique[bx].d_tRowLoc[loc_pointer];
    d_unique[bx].d_Col[d_unique[bx].point_no] =
        d_unique[bx].d_tColLoc[loc_pointer];

    // work
    ei_new = tx;
    while (ei_new < d_common.in_elem) {

      // figure out row/col location in new matrix
      row = (ei_new + 1) % d_common.in_rows - 1;     // (0-n) row
      col = (ei_new + 1) / d_common.in_rows + 1 - 1; // (0-n) column
      if ((ei_new + 1) % d_common.in_rows == 0) {
        row = d_common.in_rows - 1;
        col = col - 1;
      }

      // figure out row/col location in corresponding new template area in image
      // and give to every thread (get top left corner and progress down and
      // right)
      ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
      ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
      ori_pointer = ori_col * d_common.frame_rows + ori_row;

      // update template
      d_in[ei_new] =
          d_common.alpha * d_in[ei_new] +
          (1.00 - d_common.alpha) * d_common_change.d_frame[ori_pointer];

      // go for second round
      ei_new = ei_new + NUMBER_THREADS;
    }
  }
}

#define WARMUP
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
std::chrono::high_resolution_clock::time_point start_warmup;
std::chrono::high_resolution_clock::time_point end_warmup;

void write_data(char *filename, int frameNo, int frames_processed,
                int endoPoints, int *input_a, int *input_b, int epiPoints,
                int *input_2a, int *input_2b) {
  // printf("Writing output to %s\n", filename);
  FILE *fid;
  int i, j;
  char c;

  fid = fopen(filename, "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return;
  }

  fprintf(fid, "Total AVI Frames: %d\n", frameNo);
  fprintf(fid, "Frames Processed: %d\n", frames_processed);
  fprintf(fid, "endoPoints: %d\n", endoPoints);
  fprintf(fid, "epiPoints: %d", epiPoints);
  for (j = 0; j < frames_processed; j++) {
    fprintf(fid, "\n---Frame %d---", j);
    fprintf(fid, "\n--endo--\n", j);
    for (i = 0; i < endoPoints; i++) {
      fprintf(fid, "%d\t", input_a[j + i * frameNo]);
    }
    fprintf(fid, "\n");
    for (i = 0; i < endoPoints; i++) {
      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
      fprintf(fid, "%d\t", input_b[j + i * frameNo]);
    }
    fprintf(fid, "\n--epi--\n", j);
    for (i = 0; i < epiPoints; i++) {
      // if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
      fprintf(fid, "%d\t", input_2a[j + i * frameNo]);
    }
    fprintf(fid, "\n");
    for (i = 0; i < epiPoints; i++) {
      // if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
      fprintf(fid, "%d\t", input_2b[j + i * frameNo]);
    }
  }

  fclose(fid);
}

int main(int argc, char *argv[]) {
  auto start_all = std::chrono::high_resolution_clock::now();
  // printf("WG size of kernel = %d \n", NUMBER_THREADS);

  // CUDA kernel execution parameters
  dim3 threads;
  dim3 blocks;

  // counter
  int i;
  int frames_processed;

  // frames
  char *video_file_name;
  avi_t *frames;
  fp *frame;

  if (argc != 3) {
    printf("ERROR: usage: heartwall <inputfile> <num of frames>\n");
    exit(1);
  }

  // open movie file
  video_file_name = argv[1];
  frames = (avi_t *)AVI_open_input_file(video_file_name, 1); // added casting
  if (frames == NULL) {
    AVI_print_error((char *)"Error with AVI_open_input_file");
    return -1;
  }

  // common
  common.no_frames = AVI_video_frames(frames);
  common.frame_rows = AVI_video_height(frames);
  common.frame_cols = AVI_video_width(frames);
  common.frame_elem = common.frame_rows * common.frame_cols;
  common.frame_mem = sizeof(fp) * common.frame_elem;

#ifdef WARMUP
  start_warmup = std::chrono::high_resolution_clock::now();
  // Warmup
  double *warm;
  hipMalloc((void **)&warm, sizeof(double) * 100000);
  hipStream_t stream;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipFree(warm);
  end_warmup = std::chrono::high_resolution_clock::now();
#endif
  s_compute = std::chrono::high_resolution_clock::now();
  // pointers
  hipMalloc((void **)&common_change.d_frame, common.frame_mem);

  frames_processed = atoi(argv[2]);
  if (frames_processed < 0 || frames_processed > common.no_frames) {
    printf("ERROR: %d is an incorrect number of frames specified, select in "
           "the range of 0-%d\n",
           frames_processed, common.no_frames);
    return 0;
  }

  common.sSize = 40;
  common.tSize = 25;
  common.maxMove = 10;
  common.alpha = 0.87;

  common.endoPoints = ENDO_POINTS;
  common.endo_mem = sizeof(int) * common.endoPoints;

  common.endoRow = (int *)malloc(common.endo_mem);
  common.endoRow[0] = 369;
  common.endoRow[1] = 400;
  common.endoRow[2] = 429;
  common.endoRow[3] = 452;
  common.endoRow[4] = 476;
  common.endoRow[5] = 486;
  common.endoRow[6] = 479;
  common.endoRow[7] = 458;
  common.endoRow[8] = 433;
  common.endoRow[9] = 404;
  common.endoRow[10] = 374;
  common.endoRow[11] = 346;
  common.endoRow[12] = 318;
  common.endoRow[13] = 294;
  common.endoRow[14] = 277;
  common.endoRow[15] = 269;
  common.endoRow[16] = 275;
  common.endoRow[17] = 287;
  common.endoRow[18] = 311;
  common.endoRow[19] = 339;
  hipMalloc((void **)&common.d_endoRow, common.endo_mem);
  hipMemcpy(common.d_endoRow, common.endoRow, common.endo_mem,
             hipMemcpyHostToDevice);

  common.endoCol = (int *)malloc(common.endo_mem);
  common.endoCol[0] = 408;
  common.endoCol[1] = 406;
  common.endoCol[2] = 397;
  common.endoCol[3] = 383;
  common.endoCol[4] = 354;
  common.endoCol[5] = 322;
  common.endoCol[6] = 294;
  common.endoCol[7] = 270;
  common.endoCol[8] = 250;
  common.endoCol[9] = 237;
  common.endoCol[10] = 235;
  common.endoCol[11] = 241;
  common.endoCol[12] = 254;
  common.endoCol[13] = 273;
  common.endoCol[14] = 300;
  common.endoCol[15] = 328;
  common.endoCol[16] = 356;
  common.endoCol[17] = 383;
  common.endoCol[18] = 401;
  common.endoCol[19] = 411;
  hipMalloc((void **)&common.d_endoCol, common.endo_mem);
  hipMemcpy(common.d_endoCol, common.endoCol, common.endo_mem,
             hipMemcpyHostToDevice);

  common.tEndoRowLoc = (int *)malloc(common.endo_mem * common.no_frames);
  hipMalloc((void **)&common.d_tEndoRowLoc,
             common.endo_mem * common.no_frames);

  common.tEndoColLoc = (int *)malloc(common.endo_mem * common.no_frames);
  hipMalloc((void **)&common.d_tEndoColLoc,
             common.endo_mem * common.no_frames);

  common.epiPoints = EPI_POINTS;
  common.epi_mem = sizeof(int) * common.epiPoints;

  common.epiRow = (int *)malloc(common.epi_mem);
  common.epiRow[0] = 390;
  common.epiRow[1] = 419;
  common.epiRow[2] = 448;
  common.epiRow[3] = 474;
  common.epiRow[4] = 501;
  common.epiRow[5] = 519;
  common.epiRow[6] = 535;
  common.epiRow[7] = 542;
  common.epiRow[8] = 543;
  common.epiRow[9] = 538;
  common.epiRow[10] = 528;
  common.epiRow[11] = 511;
  common.epiRow[12] = 491;
  common.epiRow[13] = 466;
  common.epiRow[14] = 438;
  common.epiRow[15] = 406;
  common.epiRow[16] = 376;
  common.epiRow[17] = 347;
  common.epiRow[18] = 318;
  common.epiRow[19] = 291;
  common.epiRow[20] = 275;
  common.epiRow[21] = 259;
  common.epiRow[22] = 256;
  common.epiRow[23] = 252;
  common.epiRow[24] = 252;
  common.epiRow[25] = 257;
  common.epiRow[26] = 266;
  common.epiRow[27] = 283;
  common.epiRow[28] = 305;
  common.epiRow[29] = 331;
  common.epiRow[30] = 360;
  hipMalloc((void **)&common.d_epiRow, common.epi_mem);
  hipMemcpy(common.d_epiRow, common.epiRow, common.epi_mem,
             hipMemcpyHostToDevice);

  common.epiCol = (int *)malloc(common.epi_mem);
  common.epiCol[0] = 457;
  common.epiCol[1] = 454;
  common.epiCol[2] = 446;
  common.epiCol[3] = 431;
  common.epiCol[4] = 411;
  common.epiCol[5] = 388;
  common.epiCol[6] = 361;
  common.epiCol[7] = 331;
  common.epiCol[8] = 301;
  common.epiCol[9] = 273;
  common.epiCol[10] = 243;
  common.epiCol[11] = 218;
  common.epiCol[12] = 196;
  common.epiCol[13] = 178;
  common.epiCol[14] = 166;
  common.epiCol[15] = 157;
  common.epiCol[16] = 155;
  common.epiCol[17] = 165;
  common.epiCol[18] = 177;
  common.epiCol[19] = 197;
  common.epiCol[20] = 218;
  common.epiCol[21] = 248;
  common.epiCol[22] = 276;
  common.epiCol[23] = 304;
  common.epiCol[24] = 333;
  common.epiCol[25] = 361;
  common.epiCol[26] = 391;
  common.epiCol[27] = 415;
  common.epiCol[28] = 434;
  common.epiCol[29] = 448;
  common.epiCol[30] = 455;
  hipMalloc((void **)&common.d_epiCol, common.epi_mem);
  hipMemcpy(common.d_epiCol, common.epiCol, common.epi_mem,
             hipMemcpyHostToDevice);

  common.tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
  hipMalloc((void **)&common.d_tEpiRowLoc, common.epi_mem * common.no_frames);

  common.tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);
  hipMalloc((void **)&common.d_tEpiColLoc, common.epi_mem * common.no_frames);

  common.allPoints = ALL_POINTS;

  // common
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in_mem = sizeof(fp) * common.in_elem;

  // common
  hipMalloc((void **)&common.d_endoT, common.in_mem * common.endoPoints);
  hipMalloc((void **)&common.d_epiT, common.in_mem * common.epiPoints);

  for (i = 0; i < common.endoPoints; i++) {
    unique[i].point_no = i;
    unique[i].d_Row = common.d_endoRow;
    unique[i].d_Col = common.d_endoCol;
    unique[i].d_tRowLoc = common.d_tEndoRowLoc;
    unique[i].d_tColLoc = common.d_tEndoColLoc;
    unique[i].d_T = common.d_endoT;
  }
  for (i = common.endoPoints; i < common.allPoints; i++) {
    unique[i].point_no = i - common.endoPoints;
    unique[i].d_Row = common.d_epiRow;
    unique[i].d_Col = common.d_epiCol;
    unique[i].d_tRowLoc = common.d_tEpiRowLoc;
    unique[i].d_tColLoc = common.d_tEpiColLoc;
    unique[i].d_T = common.d_epiT;
  }

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    unique[i].in_pointer = unique[i].point_no * common.in_elem;
  }

  // common
  common.in2_rows = 2 * common.sSize + 1;
  common.in2_cols = 2 * common.sSize + 1;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(float) * common.in2_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2, common.in2_mem);
  }

  // common
  common.conv_rows =
      common.in_rows + common.in2_rows - 1; // number of rows in I
  common.conv_cols =
      common.in_cols + common.in2_cols - 1; // number of columns in I
  common.conv_elem = common.conv_rows * common.conv_cols; // number of elements
  common.conv_mem = sizeof(float) * common.conv_elem;
  common.ioffset = 0;
  common.joffset = 0;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_conv, common.conv_mem);
  }

  // common
  common.in2_pad_add_rows = common.in_rows;
  common.in2_pad_add_cols = common.in_cols;

  common.in2_pad_cumv_rows = common.in2_rows + 2 * common.in2_pad_add_rows;
  common.in2_pad_cumv_cols = common.in2_cols + 2 * common.in2_pad_add_cols;
  common.in2_pad_cumv_elem =
      common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_mem = sizeof(float) * common.in2_pad_cumv_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_pad_cumv, common.in2_pad_cumv_mem);
  }

  // common
  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows; // (1 to n+1)
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rows =
      common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
  common.in2_pad_cumv_sel_cols =
      common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
  common.in2_pad_cumv_sel_elem =
      common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel_mem = sizeof(float) * common.in2_pad_cumv_sel_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_pad_cumv_sel,
               common.in2_pad_cumv_sel_mem);
  }
  // common
  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig =
      common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_rows =
      common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
  common.in2_sub_cumh_cols =
      common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
  common.in2_sub_cumh_elem =
      common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_mem = sizeof(float) * common.in2_sub_cumh_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_sub_cumh, common.in2_sub_cumh_mem);
  }

  // common
  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_rows =
      common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
  common.in2_sub_cumh_sel_cols =
      common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
  common.in2_sub_cumh_sel_elem =
      common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel_mem = sizeof(float) * common.in2_sub_cumh_sel_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_sub_cumh_sel,
               common.in2_sub_cumh_sel_mem);
  }

  // common
  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig =
      common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_rows =
      common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
  common.in2_sub2_cols =
      common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sub2_mem = sizeof(float) * common.in2_sub2_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_sub2, common.in2_sub2_mem);
  }

  // common
  common.in2_sqr_rows = common.in2_rows;
  common.in2_sqr_cols = common.in2_cols;
  common.in2_sqr_elem = common.in2_elem;
  common.in2_sqr_mem = common.in2_mem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_sqr, common.in2_sqr_mem);
  }
  // common
  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in2_sqr_sub2, common.in2_sqr_sub2_mem);
  }

  // common
  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_in_sqr, common.in_sqr_mem);
  }

  // common
  common.tMask_rows = common.in_rows + (common.sSize + 1 + common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem = sizeof(float) * common.tMask_elem;

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_tMask, common.tMask_mem);
  }

  // common
  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem = sizeof(float) * common.mask_elem;
  // common
  common.mask_conv_rows = common.tMask_rows; // number of rows in I
  common.mask_conv_cols = common.tMask_cols; // number of columns in I
  common.mask_conv_elem =
      common.mask_conv_rows * common.mask_conv_cols; // number of elements
  common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
  common.mask_conv_ioffset = (common.mask_rows - 1) / 2;
  if ((common.mask_rows - 1) % 2 > 0.5) {
    common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
  }
  common.mask_conv_joffset = (common.mask_cols - 1) / 2;
  if ((common.mask_cols - 1) % 2 > 0.5) {
    common.mask_conv_joffset = common.mask_conv_joffset + 1;
  }

  // pointers
  for (i = 0; i < common.allPoints; i++) {
    hipMalloc((void **)&unique[i].d_mask_conv, common.mask_conv_mem);
  }

  // All kernels operations within kernel use same max size of threads. Size of
  // block size is set to the size appropriate for max size operation (on padded
  // matrix). Other use subsets of that.
  threads.x = NUMBER_THREADS; // define the number of threads in the block
  threads.y = 1;
  blocks.x = common.allPoints; // define the number of blocks in the grid
  blocks.y = 1;

  hipMemcpyToSymbol(HIP_SYMBOL(d_common), &common, sizeof(params_common));
  hipMemcpyToSymbol(HIP_SYMBOL(d_unique), &unique, sizeof(params_unique) * ALL_POINTS);

  for (common_change.frame_no = 0; common_change.frame_no < frames_processed;
       common_change.frame_no++) {

    // Extract a cropped version of the first frame from the video file
    frame = get_frame(
        frames,                 // pointer to video file
        common_change.frame_no, // number of frame that needs to be returned
        0,                      // cropped?
        0,                      // scaled?
        1);                     // converted

    // copy frame to GPU memory
    hipMemcpy(common_change.d_frame, frame, common.frame_mem,
               hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(d_common_change), &common_change,
                       sizeof(params_common_change));

    // launch GPU kernel
    kernel<<<blocks, threads>>>();

    // free frame after each loop iteration, since AVI library allocates memory
    // for every frame fetched
    free(frame);
  }

  hipMemcpy(common.tEndoRowLoc, common.d_tEndoRowLoc,
             common.endo_mem * common.no_frames, hipMemcpyDeviceToHost);
  hipMemcpy(common.tEndoColLoc, common.d_tEndoColLoc,
             common.endo_mem * common.no_frames, hipMemcpyDeviceToHost);

  hipMemcpy(common.tEpiRowLoc, common.d_tEpiRowLoc,
             common.epi_mem * common.no_frames, hipMemcpyDeviceToHost);
  hipMemcpy(common.tEpiColLoc, common.d_tEpiColLoc,
             common.epi_mem * common.no_frames, hipMemcpyDeviceToHost);

  // frame
  hipFree(common_change.d_frame);
  hipFree(common.d_endoRow);
  hipFree(common.d_endoCol);
  hipFree(common.d_tEndoRowLoc);
  hipFree(common.d_tEndoColLoc);
  hipFree(common.d_endoT);
  hipFree(common.d_epiRow);
  hipFree(common.d_epiCol);
  hipFree(common.d_tEpiRowLoc);
  hipFree(common.d_tEpiColLoc);
  hipFree(common.d_epiT);

  for (i = 0; i < common.allPoints; i++) {
    hipFree(unique[i].d_in2);

    hipFree(unique[i].d_conv);
    hipFree(unique[i].d_in2_pad_cumv);
    hipFree(unique[i].d_in2_pad_cumv_sel);
    hipFree(unique[i].d_in2_sub_cumh);
    hipFree(unique[i].d_in2_sub_cumh_sel);
    hipFree(unique[i].d_in2_sub2);
    hipFree(unique[i].d_in2_sqr);
    hipFree(unique[i].d_in2_sqr_sub2);
    hipFree(unique[i].d_in_sqr);

    hipFree(unique[i].d_tMask);
    hipFree(unique[i].d_mask_conv);
  }
  e_compute = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> compute_milli =
      e_compute - s_compute;
  std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

#ifdef WARMUP
  std::chrono::duration<double, std::milli> elapsed_milli_warmup =
      end_warmup - start_warmup;
  std::cerr << "Warmup time: " << elapsed_milli_warmup.count() << " ms"
            << std::endl;
#endif
  write_data("result.txt", common.no_frames, frames_processed,
             common.endoPoints, common.tEndoRowLoc, common.tEndoColLoc,
             common.epiPoints, common.tEpiRowLoc, common.tEpiColLoc);
  // epi points
  free(common.epiRow);
  free(common.epiCol);
  free(common.tEpiRowLoc);
  free(common.tEpiColLoc);
  free(common.endoRow);
  free(common.endoCol);
  free(common.tEndoRowLoc);
  free(common.tEndoColLoc);
  auto end_all = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli = end_all - start_all;
  std::cerr << "Elapsed time: " << elapsed_milli.count() << " ms" << std::endl;
}
