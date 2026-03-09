# -----------------------------
# Configure (provided by buildall.sh)
# -----------------------------
CUDA_DIR       ?= /usr/local/cuda
CUDA_LIB_DIR   ?=
GENCODE_FLAGS  ?=
CXXFLAGS       ?=

EXECUTABLE := dwt2d

# Compilers
CXX  := g++
CC   := gcc
NVCC ?= nvcc

# Files
CFILES :=
CXXFILES :=
CUFILES := main.cu dwt.cu components.cu \
           dwt_cuda/fdwt53.cu dwt_cuda/fdwt97.cu dwt_cuda/common.cu \
           dwt_cuda/rdwt97.cu dwt_cuda/rdwt53.cu

# -----------------------------
# Includes / libs
# -----------------------------
INCLUDES := -I. -I$(CUDA_DIR)/include
ifneq ($(strip $(INCLUDE)),)
INCLUDES += -I$(INCLUDE)
endif

LIBDIRS :=
ifneq ($(strip $(CUDA_LIB_DIR)),)
LIBDIRS += -L$(CUDA_LIB_DIR)
endif
LIBDIRS += -L$(CUDA_DIR)/lib64

# -----------------------------
# Flags
# -----------------------------
HOST_CXXFLAGS := $(CXXFLAGS) $(INCLUDES) $(LIBDIRS)
HOST_CFLAGS   := $(INCLUDES) $(LIBDIRS) -std=c99

NVCCFLAGS := $(INCLUDES) $(LIBDIRS) -O3 --compiler-options -fno-strict-aliasing

OUTPUT ?= -DOUTPUT

LDFLAGS := $(LIBDIRS)

# Objects
COBJS   = $(CFILES:.c=.c.o)
CXXOBJS = $(CXXFILES:.cpp=.cpp.o)
CUOBJS  = $(CUFILES:.cu=.cu.o)

.SUFFIXES: .c.o .cpp.o .cu.o .cu

%.c.o: %.c
	$(CC) $(HOST_CFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(HOST_CXXFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(OUTPUT) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

$(EXECUTABLE): $(COBJS) $(CXXOBJS) $(CUOBJS)
	$(NVCC) $(GENCODE_FLAGS) -o $(EXECUTABLE) $(COBJS) $(CXXOBJS) $(CUOBJS) $(LDFLAGS)

clean:
	@rm -f $(COBJS) $(CXXOBJS) $(CUOBJS) $(EXECUTABLE) *.bmp.dwt.* *.csv
