C_C = gcc
SRC = pathfinder_hip.cpp
EXE = pathfinder
release:
	hipcc $(CXXFLAGS) -fstrict-aliasing $(SRC) -o $(EXE) 

clean: 
	rm -f	*.o $(EXE) *.csv result.txt
