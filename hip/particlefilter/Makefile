SRC = ex_particle_CUDA_float_seq.cpp
EXE = particlefilter_float

all: float

float: $(SRC)
	hipcc $(CXXFLAGS) -O3 $(SRC) -o $(EXE)
clean:
	rm particlefilter_naive particlefilter_float *.csv *.txt
