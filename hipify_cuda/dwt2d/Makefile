#Configure
#CUDA_INSTALL_PATH ?= /usr/local/cuda
#CUDA_INSTALL_PATH = $(CUDA_DIR)
EXECUTABLE := dwt2d


# Files
CFILES := 
CXXFILES := 
CUFILES := main_hip.cpp dwt_hip.cpp components_hip.cpp dwt_cuda/fdwt53_hip.cpp dwt_cuda/fdwt97_hip.cpp dwt_cuda/common_hip.cpp dwt_cuda/rdwt97_hip.cpp dwt_cuda/rdwt53_hip.cpp

# Includes
INCLUDES := -I.

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -std=c99  
#LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64

# Warning flags (from cuda common.mk)
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CFLAGS += $(CWARN_FLAGS)
CXXFLAGS += $(CXXWARN_FLAGS)

COMMONFLAGS += -O3
NVCCFLAGS   += --compiler-options -fno-strict-aliasing
CXXFLAGS    += -fno-strict-aliasing
CFLAGS      += -fno-strict-aliasing

OUTPUT = -DOUTPUT

# Compilers
CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := hipcc

# Generate object files list
COBJS=$(CFILES:.c=.c.o)
CXXOBJS=$(CXXFILES:.cpp=.cpp.o)
CUOBJS=$(CUFILES:.cu=.cu.o)

.SUFFIXES: .c.o .cpp.o .cu.o .cu 

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(OUTPUT) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(COBJS) $(CXXOBJS) $(CUOBJS) 
	$(NVCC) -o $(EXECUTABLE) $(COBJS) $(CXXOBJS) $(CUOBJS) $(LDFLAGS)

clean:
	rm -f $(COBJS) $(CXXOBJS) $(CUOBJS) $(EXECUTABLE)
	rm *.bmp.dwt.* *.csv
