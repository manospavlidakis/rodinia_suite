#include ../../common/make.config

ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/device/device.o
	$(CC) $(CXXFLAGS)	$(KERNEL_DIM) main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/device/device.o \
			-lm \
			-L$(CUDA_LIB_DIR) \
			-lcuda -lcudart \
			-O3\
			-o lavaMD

# compile function files into objects (binaries)
main.o:		main.h \
			main.cu \
			./kernel/kernel_gpu_cuda_wrapper.h \
			./kernel/kernel_gpu_cuda_wrapper.cu \
			./util/device/device.h \
			./util/device/device.cu
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(OUTPUT) main.cu \
			-c \
			-o main.o \
			-O3\
			-std=c++11\
			-lcuda -lcudart\
			-L$(CUDA_LIB_DIR)

./kernel/kernel_gpu_cuda_wrapper.o:	./kernel/kernel_gpu_cuda_wrapper.h \
									./kernel/kernel_gpu_cuda_wrapper.cu
	$(CC) $(KERNEL_DIM)						./kernel/kernel_gpu_cuda_wrapper.cu \
									-c \
									-o ./kernel/kernel_gpu_cuda_wrapper.o \
									-O3 \
									$(GENCODE_FLAGS)

./util/device/device.o:	./util/device/device.h \
						./util/device/device.cu
	$(CC)			./util/device/device.cu \
						-c \
						-o ./util/device/device.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/device/*.o \
		*.txt\
		lavaMD
