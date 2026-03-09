#include ../../common/make.config

ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

NVCC ?= nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = nw

release: $(SRC)
	$(NVCC) $(CXXFLAGS) ${KERNEL_DIM} $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum: $(SRC)
	$(NVCC) ${KERNEL_DIM} -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(NVCC) ${KERNEL_DIM} -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(NVCC) ${KERNEL_DIM} -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean: $(SRC)
	@rm -f $(EXE) $(EXE).linkinfo result.txt *.csv 8192_*
