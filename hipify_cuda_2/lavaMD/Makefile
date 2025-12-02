# C compiler
CC = g++

# link objects (binaries) together
a.out: main.o ./kernel/kernel_gpu_cuda_wrapper.o 
	hipcc $(CXXFLAGS) -O3  main.o ./kernel/kernel_gpu_cuda_wrapper.o $(CC_FLAGS) -lm -o lavaMD

# compile function files into objects (binaries)
main.o: main.h main_hip.cpp ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper_hip.cpp
	hipcc -O3 $(CXXFLAGS) $(OUTPUT) main_hip.cpp -c -o main.o

./kernel/kernel_gpu_cuda_wrapper.o: ./kernel/kernel_gpu_cuda_wrapper.h \
                                    ./kernel/kernel_gpu_cuda_wrapper_hip.cpp
	hipcc -c  ./kernel/kernel_gpu_cuda_wrapper_hip.cpp -o ./kernel/kernel_gpu_cuda_wrapper.o 

# delete all object and executable files
clean:
	rm -f *.o ./kernel/*.o lavaMD result.txt *.csv
