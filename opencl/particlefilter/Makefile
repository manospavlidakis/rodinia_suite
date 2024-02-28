OPENCL_DIR = /opt/altera/intelFPGA/20.3/hld
OPENCL_INC = $(OPENCL_DIR)/host/include/
OPENCL_LIB = $(OPENCL_DIR)/linux64/lib/

# C compiler
CC = g++
CC_FLAGS = -O3

SRC = ex_particle_OCL_single_seq.cpp
SRC1 = ex_particle_OCL_double_seq.cpp
SRC2 = ex_particle_OCL_naive_seq.cpp
EXE = ex_particle_OCL_single_seq
EXE1 = particlefilter_float
EXE2 = ex_particle_OCL_naive_seq

all: double
	#float double naive
float: $(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 
double: $(SRC1)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC1) -o $(EXE1) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 
naive: $(SRC2)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC2) -o $(EXE2) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

clean:
	rm -f *.o $(EXE) $(EXE1) $(EXE2) 
