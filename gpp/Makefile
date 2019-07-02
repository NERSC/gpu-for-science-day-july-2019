include ../Makefile.inc

SRCS    = $(shell ls *.cpp)
#CUSRCS = $(shell ls *.cu)
OBJS    = $(SRCS:.cpp=.o)
CUOBJS  = $(CUSRCS:.cu=.o)
EXEC    = gpp.ex

PROJDIR = $(shell pwd)/..
INCLS   = -I$(CUDA_ROOT)/include \
          -I$(CUDA_ROOT)/samples/common/inc \
          -I$(PROJDIR)/external \
          -I$(PROJDIR)/external/ComplexClass \
          -I$(PROJDIR)/external/arrayMD

all: gpp

gpp: $(OBJS) $(CUOBJS)
	$(CXX) $(INCLS) $(CXXFLAGS) $(OBJS) $(CUOBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) -c $(INCLS) $(CXXFLAGS) $< -o $@

%.o: %.cu
	$(CXX) -c $(INCLS) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(EXEC) *.ptx *.cub
