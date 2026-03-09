ifeq ($(d),debug)
   CXXFLAGS += -DOUTPUT
endif

NVCC ?= nvcc
INCLUDE := $(CUDA_DIR)/include

OBJ = main_test_cu.o stats_logger.o cpuencode.o
SRC = main_test_cu.cu
EXE = huffman

huffman: $(OBJ)
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) $(OBJ) -o $(EXE) -L$(CUDA_LIB_DIR)

main_test_cu.o: main_test_cu.cu cutil.h
	$(NVCC) $(GENCODE_FLAGS) -c main_test_cu.cu $(CXXFLAGS) -I$(INCLUDE)

stats_logger.o: stats_logger.cpp
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c stats_logger.cpp -o stats_logger.o -I$(INCLUDE) -L$(CUDA_LIB_DIR)

cpuencode.o: cpuencode.cpp
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -c cpuencode.cpp -o cpuencode.o -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	@rm -f *.o $(EXE) *.csv
