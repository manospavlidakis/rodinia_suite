CC = gcc

backprop: backprop.o facetrain.o imagenet.o backprop_cuda_hip.o 
	hipcc -O3 $(CXXFLAGS) backprop.o facetrain.o imagenet.o backprop_cuda_hip.o -o backprop -lm

%.o: %.[ch]
	$(CC) $(CXXFLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CXXFLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CXXFLAGS) backprop.c -c

backprop_cuda_hip.o: backprop_cuda_hip.cpp backprop.h
	hipcc $(CXXFLAGS) -c backprop_cuda_hip.cpp

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop *.csv result.txt
