# Assuming a similar debug structure and adding Output flag
ifeq ($(d),debug)
   CXXFLAGS += -DOUTPUT
endif

NVCC ?= nvcc
INCLUDE := $(CUDA_DIR)/include

# Default build target
all: cfd
	#euler3d_double pre_euler3d pre_euler3d_double

# Specific build rules
cfd: euler3d.o
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) euler3d.o -o cfd -L$(CUDA_LIB_DIR)

euler3d.o: euler3d.cu
	$(NVCC) $(CXXFLAGS) -c euler3d.cu $(GENCODE_FLAGS) -o euler3d.o -I$(INCLUDE) -L$(CUDA_LIB_DIR)

#euler3d_double: euler3d_double.o
#	$(NVCC) euler3d_double.o -o euler3d_double -L$(CUDA_LIB_DIR) -lcudart

#euler3d_double.o: euler3d_double.cu
#	$(NVCC) -Xptxas -v $(NVCC_FLAGS) -c euler3d_double.cu -o euler3d_double.o -I$(CUDA_SDK_PATH)/common/inc -I$(INCLUDE)

#pre_euler3d: pre_euler3d.o
#	$(NVCC) pre_euler3d.o -o pre_euler3d -L$(CUDA_LIB_DIR) -lcudart

#pre_euler3d.o: pre_euler3d.cu
#	$(NVCC) -Xptxas -v $(NVCC_FLAGS) -c pre_euler3d.cu -o pre_euler3d.o -I$(CUDA_SDK_PATH)/common/inc -I$(INCLUDE)

#pre_euler3d_double: pre_euler3d_double.o
#	$(NVCC) pre_euler3d_double.o -o pre_euler3d_double -L$(CUDA_LIB_DIR) -lcudart

#pre_euler3d_double.o: pre_euler3d_double.cu
#	$(NVCC) -Xptxas -v $(NVCC_FLAGS) -c pre_euler3d_double.cu -o pre_euler3d_double.o -I$(CUDA_SDK_PATH)/common/inc -I$(INCLUDE)

# Clean rule
clean:
	@rm -f cfd *.o *.linkinfo result_density.txt result_momentum.txt result_density_energy.txt
