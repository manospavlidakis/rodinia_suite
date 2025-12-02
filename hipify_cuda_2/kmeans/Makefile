CC := $(CUDA_DIR)/bin/nvcc
CXXFLAGS += -DOUTPUT
INCLUDE := $(CUDA_DIR)/include

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(CC) $(CXXFLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lm

%.o: %.[ch]
	$(CC) $(CXXFLAGS) $< -c  -O3 -I$(INCLUDE) -L$(CUDA_LIB_DIR)

kmeans.o: kmeans.cu
	$(CC) $(CXXFLAGS) -c kmeans.cu  -O3 -I$(INCLUDE) -L$(CUDA_LIB_DIR)


kmeans_cuda.o: kmeans_cuda.cu
	$(CC) $(CXXFLAGS) -c kmeans_cuda.cu  -O3 -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	@rm -f *.o *~ kmeans kmeans_cuda.linkinfo result.txt
