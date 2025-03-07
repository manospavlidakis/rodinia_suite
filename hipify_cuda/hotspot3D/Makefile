SRC = 3D_hip.cpp

EXE = hotspot3D

release: $(SRC)
	hipcc -O3 $(CXXFLAGS) $(SRC) -o $(EXE) 

clean:
	rm -rf $(EXE) $(EXE).linkinfo $(OUTPUT) *.csv 512x4_1000* result.txt output
