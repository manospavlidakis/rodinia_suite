#include ../../common/make.config

ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(CC) $(CXXFLAGS) $(KERNEL_DIM) $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
