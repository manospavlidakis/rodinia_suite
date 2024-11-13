# C compiler
CC = g++

# link objects (binaries) together
a.out: main.o ./kernel/kernel_gpu_cuda_wrapper.o 
	hipcc $(CXXFLAGS) -O3  main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			$(CC_FLAGS) -lm \
			-o lavaMD

# compile function files into objects (binaries)
main.o: main.h main.cpp ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper.cu 
	hipcc $(CXXFLAGS) $(OUTPUT) main.cpp -c -o main.o -O3

./kernel/kernel_gpu_cuda_wrapper.o: ./kernel/kernel_gpu_cuda_wrapper.h \
                                    ./kernel/kernel_gpu_cuda_wrapper.cu
	hipcc -c  ./kernel/kernel_gpu_cuda_wrapper.cu -o ./kernel/kernel_gpu_cuda_wrapper.o 

# delete all object and executable files
clean:
	rm -f *.o ./kernel/*.o lavaMD result.txt
