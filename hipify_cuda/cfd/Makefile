CC := gcc

all: cfd

# Specific build rules
cfd: euler3d.o
	hipcc -O3 $(CXXFLAGS) -D_MY_IS_HIP  euler3d.o -o cfd 
euler3d.o: euler3d_hip.cpp
	hipcc -O3 $(CXXFLAGS) -D_MY_IS_HIP -c euler3d_hip.cpp $(GENCODE_FLAGS) -o euler3d.o

#euler3d_double: euler3d_double.o
#	$(CC) euler3d_double.o -o euler3d_double -L$(CUDA_LIB_DIR) -lcudart

#euler3d_double.o: euler3d_double.cu
#	$(CC) -Xptxas -v $(NVCC_FLAGS) -c euler3d_double.cu -o euler3d_double.o -I$(CUDA_SDK_PATH)/common/inc -I$(INCLUDE)

#pre_euler3d: pre_euler3d.o
#	$(CC) pre_euler3d.o -o pre_euler3d -L$(CUDA_LIB_DIR) -lcudart

#pre_euler3d.o: pre_euler3d.cu
#	$(CC) -Xptxas -v $(NVCC_FLAGS) -c pre_euler3d.cu -o pre_euler3d.o -I$(CUDA_SDK_PATH)/common/inc -I$(INCLUDE)

#pre_euler3d_double: pre_euler3d_double.o
#	$(CC) pre_euler3d_double.o -o pre_euler3d_double -L$(CUDA_LIB_DIR) -lcudart

#pre_euler3d_double.o: pre_euler3d_double.cu
#	$(CC) -Xptxas -v $(NVCC_FLAGS) -c pre_euler3d_double.cu -o pre_euler3d_double.o -I$(CUDA_SDK_PATH)/common/inc -I$(INCLUDE)

# Clean rule
clean:
	rm -f cfd *.o *.linkinfo result_density.txt result_momentum.txt result_density_energy.txt

