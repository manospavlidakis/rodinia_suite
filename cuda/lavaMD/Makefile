ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

# link objects (binaries) together
a.out:	main.o ./kernel/kernel_gpu_cuda_wrapper.o ./util/device/device.o
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) main.o ./kernel/kernel_gpu_cuda_wrapper.o 	./util/device/device.o -lm -L$(CUDA_LIB_DIR) -O3 -o lavaMD

# compile function files into objects (binaries)
main.o:	main.h main.cu ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper.cu \
			./util/device/device.h ./util/device/device.cu
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) $(OUTPUT) main.cu -c -o main.o -O3	-std=c++11 -L$(CUDA_LIB_DIR)

./kernel/kernel_gpu_cuda_wrapper.o:	./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper.cu
	$(CC) $(KERNEL_DIM)	$(GENCODE_FLAGS) ./kernel/kernel_gpu_cuda_wrapper.cu -c -o ./kernel/kernel_gpu_cuda_wrapper.o -O3

./util/device/device.o:	./util/device/device.h ./util/device/device.cu
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./util/device/device.cu -c -o ./util/device/device.o -O3

# delete all object and executable files
clean:
	@rm -f	*.o ./kernel/*.o ./util/device/*.o result.txt lavaMD *.csv
