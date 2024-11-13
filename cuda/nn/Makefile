#include ../../common/make.config

INCLUDE := $(CUDA_DIR)/include

COMPILER_PATH := /usr/bin

ifeq ($(d),debug)
    CXXFLAGS += -DOUTPUT
endif

# Compilers
CC         := $(COMPILER_PATH)/gcc
CXX        := $(COMPILER_PATH)/g++
LINK       := $(COMPILER_PATH)/g++ -fPIC
NVCC       := $(CUDA_DIR)/bin/nvcc

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen 2048_*_nn.csv nat_result.txt

nn : nn_cuda.cu
	$(NVCC) $(CXXFLAGS) $(KERNEL_DIM) $(GENCODE_FLAGS) -o $@ $<

hurricane_gen : hurricane_gen.c
	$(CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
