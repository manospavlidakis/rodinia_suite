CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(CC) $(CXXFLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop -L$(CUDA_LIB_DIR) -lm

%.o: %.[ch]
	$(CC) $(CXXFLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CXXFLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CXXFLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(CC) $(CXXFLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo *.csv result.txt
