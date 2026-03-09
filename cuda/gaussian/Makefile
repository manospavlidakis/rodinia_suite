#include ../../common/make.config
ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

NVCC ?= nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(NVCC) $(CXXFLAGS) $(KERNEL_DIM) $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	@rm -f gaussian *.csv result.txt 2048_*
