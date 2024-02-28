include ../../common/make.config

CUDA_INSTALL_PATH ?= /usr/local/cuda
ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

INCLUDE := $(CUDA_INSTALL_PATH)/include
COMPILER_PATH := /usr/bin

# Compilers
C_C         := $(COMPILER_PATH)/gcc
CXX        := $(COMPILER_PATH)/g++
LINK       := $(COMPILER_PATH)/g++ -fPIC
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CXX)

SRC = ex_particle_CUDA_float_seq.cu
EXE = particlefilter_float
# Compilers
CC := $(NVCC)

all: float

float: $(SRC)
	$(CC) $(CXXFLAGS) $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm particlefilter_naive particlefilter_float
