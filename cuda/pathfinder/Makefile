#include ../../common/make.config
ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

INCLUDE := $(CUDA_DIR)/include

# Compilers
C_C  := gcc
CXX  := g++
LINK := g++ -fPIC
NVCC ?= nvcc

SRC = pathfinder.cu

EXE = pathfinder

release:
	$(NVCC) $(CXXFLAGS) $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum:
	$(NVCC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug:
	$(NVCC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum:
	$(NVCC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	@rm -f pathfinder *.csv result.txt
