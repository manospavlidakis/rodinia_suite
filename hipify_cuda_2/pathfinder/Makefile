C_C = gcc
SRC = pathfinder_hip.cpp
EXE = pathfinder
release:
	hipcc -O3 $(CXXFLAGS) $(SRC) -o $(EXE) 

clean: 
	rm -f	*.o $(EXE) *.csv result.txt
