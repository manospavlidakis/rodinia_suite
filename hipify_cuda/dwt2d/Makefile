# ---- dwt2d (HIP) ----

EXECUTABLE := dwt2d

# Sources
HIP_SRCS := \
  main_hip.cpp \
  dwt_hip.cpp \
  components_hip.cpp \
  dwt_cuda/fdwt53_hip.cpp \
  dwt_cuda/fdwt97_hip.cpp \
  dwt_cuda/common_hip.cpp \
  dwt_cuda/rdwt97_hip.cpp \
  dwt_cuda/rdwt53_hip.cpp

INCLUDES := -I.

# Compiler/linker
CXX      := hipcc
CXXFLAGS := -O3 -fno-strict-aliasing $(INCLUDES) \
            -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts \
            -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith \
            -Wcast-align -Wreturn-type -Wno-unused-function
LDFLAGS  :=

OBJS := $(HIP_SRCS:.cpp=.o)

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXECUTABLE)
	rm -f *.bmp.dwt.* *.csv

