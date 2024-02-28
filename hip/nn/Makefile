C_C = gcc
SRC = nn_hip.cpp
EXE = nn

release: $(SRC)
	hipcc -O3 $(SRC) -o $(EXE) 

clean:
	rm -f *.o $(EXE) 
