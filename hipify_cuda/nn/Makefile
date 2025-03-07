C_C = gcc
SRC = nn_cuda_hip.cpp
EXE = nn

release: $(SRC)
	hipcc $(CXXFLAGS) -fstrict-aliasing $(SRC) -o $(EXE) 

clean:
	rm -f *.o $(EXE) 2048_*_nn.csv result.txt
