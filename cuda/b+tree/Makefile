CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

b+tree:	./main.o ./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o ./util/num/num.o ./util/cuda/cuda.o
		$(CC) $(CXXFLAGS) $(GENCODE_FLAGS) ./main.o ./kernel/kernel_gpu_cuda_wrapper.o \
			./kernel/kernel_gpu_cuda_wrapper_2.o ./util/num/num.o ./util/cuda/cuda.o -lm \
			-L$(CUDA_LIB_DIR) -O3 -o b+tree

main.o:	./common.h ./main.h ./main.cu
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./main.cu -c -o ./main.o -O3 -L$(CUDA_LIB_DIR)

./kernel/kernel_gpu_cuda_wrapper.o:	./common.h ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper.cu

	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./kernel/kernel_gpu_cuda_wrapper.cu -c -o ./kernel/kernel_gpu_cuda_wrapper.o -O3

./kernel/kernel_gpu_cuda_wrapper_2.o: ./common.h ./kernel/kernel_gpu_cuda_wrapper_2.h ./kernel/kernel_gpu_cuda_wrapper_2.cu
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) ./kernel/kernel_gpu_cuda_wrapper_2.cu -c -o ./kernel/kernel_gpu_cuda_wrapper_2.o -O3


./util/num/num.o: ./common.h ./util/num/num.h ./util/num/num.c
	$(CC) $(GENCODE_FLAGS) ./util/num/num.c -c -o ./util/num/num.o -O3

./util/cuda/cuda.o:	./common.h ./util/cuda/cuda.h ./util/cuda/cuda.cu
	$(CC) $(GENCODE_FLAGS) ./util/cuda/cuda.cu -c -o ./util/cuda/cuda.o -O3

clean:
	@rm -f	*.o ./kernel/*.o ./util/num/*.o ./util/cuda/*.o  result.txt b+tree *csv
