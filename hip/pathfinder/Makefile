C_C = gcc
SRC = pathfinder.cpp
EXE = pathfinder
release:
	hipcc $(CXXFLAGS) -O3 $(SRC) -o $(EXE) 

clean: 
	rm -f	*.o $(EXE) *.csv *.txt
