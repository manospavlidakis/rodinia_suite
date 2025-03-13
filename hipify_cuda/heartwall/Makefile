CC = gcc

# link objects(binaries) together
heartwall: main_hip.o ./AVI/avilib.o ./AVI/avimod.o 
	hipcc -O3 $(CXXFLAGS)  main_hip.o ./AVI/avilib.o ./AVI/avimod.o  -lm -o heartwall

# compile main function file into object (binary)
main_hip.o: main_hip.cpp  define.c
	hipcc -O3  $(CXXFLAGS) main_hip.cpp -I./AVI -c -O3

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo result.txt
