# C compiler
CC = gcc
SRC = gaussian.cu
EXE = gaussian
release: $(SRC)
	hipcc -O3 $(CXXFLAGS) $(SRC) -o $(EXE) 

clean:
	rm -f *.o $(EXE) 
