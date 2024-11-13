#include ../../common/make.config
#include ../../common/make.config
ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = ex_particle_CUDA_float_seq.cu
EXE = particlefilter_float

all: float

float: $(SRC)
	$(CC) $(CXXFLAGS) $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)
clean:
	rm particlefilter_naive particlefilter_float *.csv *.txt
