ifeq ($(d),debug)
   CXXFLAGS += -DOUTPUT
endif
CC = g++
OBJ = main_test_cu_hip.o stats_logger.o vlc_kernel_sm64huff_hip.o scan_hip.o pack_kernels_hip.o cpuencode.o
SRC = main_test_cu_hip.cpp
EXE = huffman

huffman: $(OBJ) 
	hipcc -O3  $(CXXFLAGS) $(OBJ) -o $(EXE) 

vlc_kernel_sm64huff_hip.o:vlc_kernel_sm64huff_hip.cpp
	hipcc -O3 $(CXXFLAGS) -c vlc_kernel_sm64huff_hip.cpp

scan_hip.o: scan_hip.cpp
	hipcc -O3 -c scan_hip.cpp  $(CXXFLAGS)

pack_kernels_hip.o:pack_kernels_hip.cpp 
	hipcc -O3 -c pack_kernels_hip.cpp $(CXXFLAGS)

main_test_cu_hip.o: main_test_cu_hip.cpp cutil.h
	hipcc -O3 -c main_test_cu_hip.cpp $(CXXFLAGS) 

.o:.cpp
	$(CC) $(CXXFLAGS) -c $@ -o $< 

clean:
	rm -f *.o $(EXE) *.csv 
