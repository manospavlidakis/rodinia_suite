include ../../common/make.config

CC = g++
MAIN = hotspot
KERNEL = $(MAIN)_kernel

EXE = $(MAIN)

ifdef SSIZE
	CFLAGS += -DSSIZE=$(SSIZE)
endif

ifdef BSIZE
	CFLAGS += -DBSIZE=$(BSIZE)
	CFLAGS += -DBLOCK_X=$(BSIZE)
	CFLAGS += -DBLOCK_Y=$(BSIZE)
endif

ifdef BLOCK_X
	CFLAGS += -DBLOCK_X=$(BLOCK_X)
endif

ifdef BLOCK_Y
	CFLAGS += -DBLOCK_Y=$(BLOCK_Y)
endif

all: $(EXE)

$(EXE): hotspot.c hotspot.h hotspot_common.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CFLAGS) -o $(EXE) $(OPENCL_INC) OpenCL_helper_library.c hotspot.c $(LDFLAGS) $(OPENCL_LIB)

clean:
	$(RM) $(EXE)

