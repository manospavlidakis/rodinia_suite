OPENCL_DIR = /opt/altera/intelFPGA/20.3/hld
OPENCL_INC = $(OPENCL_DIR)/host/include/
OPENCL_LIB = $(OPENCL_DIR)/linux64/lib/

CC = g++
CC_FLAGS = -O3
EXE = nn
SRC = nearestNeighbor.cpp clutils.cpp utils.cpp

VERSIONS = 0

release: $(SRC) $(AOT_PROGRAMS)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

clean:
	rm -rf $(EXE)
