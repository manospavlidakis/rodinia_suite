include ../../common/make.config
NAME = lavaMD
EXE = $(NAME)
SRC = main.c ./kernel/kernel_gpu_opencl_wrapper.c ./util/timer/timer.c ./util/opencl/opencl.c ./util/num/num.c

VERSIONS = 0
KERNEL_NAMES = $(foreach var,$(VERSIONS),$(NAME)_kernel_v$(var))
KERNEL_ALTERA_NAMES = $(addsuffix .aocx, $(KERNEL_NAMES))

CC = g++

all: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(OPENCL_INC) $(OPENCL_LIB) $(EXTRA_FLAGS)

clean:
	$(RM) $(EXE) *.aoco $(KERNEL_NAMES)
