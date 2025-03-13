CC := $(CUDA_DIR)/bin/nvcc
CXXFLAGS += -DOUTPUT
INCLUDE := $(CUDA_DIR)/include

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o 
	$(CC) $(CXXFLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	$(CC) $(CXXFLAGS) $(GENCODE_FLAGS) main.cu -I./AVI -c -O3 -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo result.txt
