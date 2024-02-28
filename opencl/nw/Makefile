#include ../../common/make.config
OPENCL_DIR = /opt/altera/intelFPGA/20.3/hld
OPENCL_INC = $(OPENCL_DIR)/host/include/
OPENCL_LIB = $(OPENCL_DIR)/linux64/lib/

SRC = nw.c
EXE = nw

CC = g++
CC_FLAGS = -O3

all: $(SRC)
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

clean:
	$(RM) $(EXE) *.aoco
