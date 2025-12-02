# C compiler
CC = gcc
SRC = gaussian_hip.cpp
EXE = gaussian
release: $(SRC)
	hipcc -O3 $(CXXFLAGS) $(SRC) -o $(EXE) 

clean:
	rm -f *.o $(EXE) *.csv result.txt 2048_*
