CC = gcc
SRC = needle_hip.cpp

EXE = nw

release: $(SRC)
	hipcc $(CXXFLAGS) -O3 $(SRC) -o $(EXE) 

clean: 
	rm -f *.o result.txt $(EXE) *.csv 8192_*
