OPENCL_DIR = /opt/altera/intelFPGA/20.3/hld
OPENCL_INC = $(OPENCL_DIR)/host/include/
OPENCL_LIB = $(OPENCL_DIR)/linux64/lib/

#C compiler
CC         := gcc
CXX        := g++
LINK       := $(CXX) -fPIC
CC_FLAGS = -O3
EXE = pathfinder

SRC = pathfinder.cpp OpenCL.cpp

all: $(SRC)
	$(CXX) $(CC_FLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

clean:
	rm -rf $(EXE) result.txt
