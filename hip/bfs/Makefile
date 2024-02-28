# C compiler
CC = gcc
EXE = bfs
SRC = bfs.hip.cu

release: $(SRC)
	hipcc -O3 $(SRC) -o $(EXE) 

clean:
	rm -f *.o bfs result.txt
