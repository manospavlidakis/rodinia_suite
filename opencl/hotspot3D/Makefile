include ../../common/make.config
CC = g++
NAME = hotspot3D
	EXE = $(NAME)
SRC = hotspot3D.c CL_helper.c

ifdef TIME
	CFLAGS += -DTIME=$(TIME)
endif

ifdef SSIZE
	CFLAGS += -DSSIZE=$(SSIZE)
endif

ifdef BSIZE
	CFLAGS += -DBLOCK_X=$(BSIZE)
	CFLAGS += -DBLOCK_Y=$(BSIZE)
endif

ifdef BLOCK_X
	CFLAGS += -DBLOCK_X=$(BLOCK_X)
endif

ifdef BLOCK_Y
	CFLAGS += -DBLOCK_Y=$(BLOCK_Y)
endif

ifdef PAD
	CFLAGS += -DPAD=$(PAD)
endif

OUTPUT = *.txt

EXTRA_FLAGS += -lm

all: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRA_FLAGS)

clean:
	$(RM) $(EXE) *.aoco $(KERNEL_NAMES) $(OUTPUT)
