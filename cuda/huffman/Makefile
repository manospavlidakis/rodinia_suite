ifeq ($(d),debug)
   CXXFLAGS += -DOUTPUT
endif

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

OBJ = main_test_cu.o stats_logger.o cpuencode.o
SRC = main_test_cu.cu
EXE = huffman

huffman: $(OBJ)
	$(CC) $(CXXFLAGS) $(GENCODE_FLAGS) $(OBJ) -o $(EXE) -L$(CUDA_LIB_DIR)

main_test_cu.o: main_test_cu.cu cutil.h
	$(CC) $(GENCODE_FLAGS) -c main_test_cu.cu $(CXXFLAGS) -I$(INCLUDE)

.o:.cpp
	$(CC) ++  $(CXXFLAGS) $(GENCODE_FLAGS) -c $@ -o $< -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	@rm -f *.o $(EXE) *.csv
