#include ../../common/make.config
ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

INCLUDE := $(CUDA_DIR)/include

# Compilers
C_C         := gcc
CXX        := g++
LINK       := g++ -fPIC
NVCC       := $(CUDA_DIR)/bin/nvcc -ccbin $(CXX)


# Compilers
CC := $(NVCC)

#CC := $(CUDA_DIR)/bin/nvcc
#INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = pathfinder

release:
	$(NVCC) $(CXXFLAGS) $(SRC) $(GENCODE_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f pathfinder
