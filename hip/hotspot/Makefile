SRC = hotspot.cpp
EXE = hotspot

release: $(SRC)
	hipcc -O3 $(SRC) -o $(EXE) 

clean:
	rm -f *.o $(EXE) 
