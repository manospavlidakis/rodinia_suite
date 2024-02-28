#ifndef KERNEL_WRAPPER
#define KERNEL_WRAPPER
void clInit();
void clRelease();
void kernel_gpu_opencl_wrapper(par_str parms_cpu, dim_str dim_cpu,
                               box_str *box_cpu, FOUR_VECTOR *rv_cpu,
                               fp *qv_cpu, FOUR_VECTOR *fv_cpu, int version);
#endif
