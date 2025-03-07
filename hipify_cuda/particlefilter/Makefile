SRC = ex_particle_CUDA_naive_seq_hip.cpp
EXE = particlefilter_float

all: float

float: $(SRC)
	hipcc $(CXXFLAGS) -O3 $(SRC) -o $(EXE)
clean:
	rm particlefilter_naive particlefilter_float *.csv result.txt
