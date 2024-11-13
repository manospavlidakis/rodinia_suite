SRC = hotspot.cpp
EXE = hotspot

release: $(SRC)
	hipcc -O3 $(CXXFLAGS) $(SRC) -o $(EXE) 

clean:
	rm -f *.o $(EXE) *.csv 1024_* *.txt output
