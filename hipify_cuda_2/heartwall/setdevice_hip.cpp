
#include <hip/hip_runtime.h>
////////////////////////////////////////////////////////////////////////////////
// Set Device
////////////////////////////////////////////////////////////////////////////////

void setdevice(void){

	// variables
	int num_devices;
	int device;

	HIP_CHECK(hipGetDeviceCount(&num_devices));
	if (num_devices > 1) {
		
		// variables
		int max_multiprocessors; 
		int max_device;
		hipDeviceProp_t properties;

		// initialize variables
		max_multiprocessors = 0;
		max_device = 0;
		
		for (device = 0; device < num_devices; device++) {
			HIP_CHECK(hipGetDeviceProperties(&properties, device));
			if (max_multiprocessors < properties.multiProcessorCount) {
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		HIP_CHECK(hipSetDevice(max_device));
	}

}
