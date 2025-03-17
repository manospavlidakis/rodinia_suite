ifeq ($(d),debug)
   CXXFLAGS += -DOUTPUT
endif

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

OBJ = main_test_cu.o stats_logger.o vlc_kernel_sm64huff.o scan.o pack_kernels.o cpuencode.o
SRC = main_test_cu.cu
EXE = huffman

huffman: $(OBJ) 
	$(CC) $(CXXFLAGS) $(OBJ) -o $(EXE) -L$(CUDA_LIB_DIR)

vlc_kernel_sm64huff.o: vlc_kernel_sm64huff.cu 
	$(CC) $(CXXFLAGS) -c vlc_kernel_sm64huff.cu  -L$(CUDA_LIB_DIR)

scan.o: scan.cu 
	$(CC) -c scan.cu  $(CXXFLAGS)


pack_kernels.o: pack_kernels.cu 
	$(CC) -c pack_kernels.cu $(CXXFLAGS)

main_test_cu.o: main_test_cu.cu cutil.h
	$(CC) -c main_test_cu.cu $(CXXFLAGS)  -I$(INCLUDE)

.o:.cpp
	$(CC) ++  $(CXXFLAGS) -c $@ $(GENCODE_FLAGS) -o $< -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f *.o $(EXE) *.csv
