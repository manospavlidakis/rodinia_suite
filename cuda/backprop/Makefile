CC   ?= gcc
NVCC ?= nvcc
INCLUDE := $(CUDA_DIR)/include

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop -L$(CUDA_LIB_DIR) -lm

facetrain.o: facetrain.c backprop.h
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c facetrain.c -o facetrain.o

backprop.o: backprop.c backprop.h
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c backprop.c -o backprop.o

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c backprop_cuda.cu -o backprop_cuda.o

imagenet.o: imagenet.c backprop.h
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c imagenet.c -o imagenet.o

clean:
	@rm -f *.o *~ backprop backprop_cuda.linkinfo *.csv result.txt
