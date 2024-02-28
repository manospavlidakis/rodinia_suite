CC = gcc
SRC = needle_hip.cpp

EXE = nw

release: $(SRC)
	hipcc -O3 $(SRC) -o $(EXE) 

clean: 
	rm -f *.o *.txt $(EXE)
