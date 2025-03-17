#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__
#include <stdio.h>
#include <hip/hip_runtime.h>
/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/


bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	hipGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		hipDeviceProp_t prop;
		if(hipGetDeviceProperties(&prop, i) == hipSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	hipSetDevice(i);

	return true;
}
#endif
