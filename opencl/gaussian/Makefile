OPENCL_DIR = /opt/altera/intelFPGA/20.3/hld
OPENCL_INC = $(OPENCL_DIR)/host/include/
OPENCL_LIB = $(OPENCL_DIR)/linux64/lib/

#C compiler
CC = g++

SRC = clutils.cpp gaussianElim.cpp utils.cpp

CC_FLAGS = -O3

EXE = gaussian

release:
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

clean:
	rm -f $(EXE)
