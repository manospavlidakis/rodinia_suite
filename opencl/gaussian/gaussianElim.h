#ifndef _GAUSSIANELIM
#define _GAUSSIANELIM

#include <algorithm>
#include <float.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "clutils.h"

// All OpenCL headers
#if defined(__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

float *OpenClGaussianElimination(cl_context context, int timing);

void printUsage();
int parseCommandline(int argc, char *argv[], char *filename, int *q, int *t,
                     int *p, int *d, int *size);

void InitPerRun();
void ForwardSub(cl_context context, float *a, float *b, float *m, int size,
                int timing);
void BackSub();
void Fan1(float *m, float *a, int Size, int t);
void Fan2(float *m, float *a, float *b, int Size, int j1, int t);
// void Fan3(float *m, float *b, int Size, int t);
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol);
void InitAry(FILE *fp, float *ary, int ary_size);
void PrintMat(float *ary, int size, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
float eventTime(cl_event event, cl_command_queue command_queue);
#endif
