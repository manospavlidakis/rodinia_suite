CC := hipcc
INCLUDE := $(ROCM_PATH)/include

b+tree: ./main.o ./kernel/kernel_gpu_cuda_wrapper_hip.o ./kernel/kernel_gpu_cuda_wrapper_2_hip.o ./util/num/num.o
	$(CC) $(CXXFLAGS) $(GENCODE_FLAGS) ./main.o ./kernel/kernel_gpu_cuda_wrapper_hip.o \
		./kernel/kernel_gpu_cuda_wrapper_2_hip.o ./util/num/num.o -lm \
		-O3 -o b+tree

main.o: ./common.h ./main.h ./main_hip.cpp
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./main_hip.cpp -c -o ./main.o -O3

./kernel/kernel_gpu_cuda_wrapper_hip.o: ./common.h ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper_hip.cpp
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./kernel/kernel_gpu_cuda_wrapper_hip.cpp -c -o ./kernel/kernel_gpu_cuda_wrapper_hip.o -O3

./kernel/kernel_gpu_cuda_wrapper_2_hip.o: ./common.h ./kernel/kernel_gpu_cuda_wrapper_2.h ./kernel/kernel_gpu_cuda_wrapper_2_hip.cpp
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./kernel/kernel_gpu_cuda_wrapper_2_hip.cpp -c -o ./kernel/kernel_gpu_cuda_wrapper_2_hip.o -O3

./util/num/num.o: ./common.h ./util/num/num.h ./util/num/num.c
	$(CC) $(GENCODE_FLAGS) ./util/num/num.c -c -o ./util/num/num.o -O3

clean:
	@rm -f *.o ./kernel/*.o ./util/num/*.o result.txt b+tree *csv a.out
