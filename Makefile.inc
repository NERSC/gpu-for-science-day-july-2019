ifeq ($(findstring openmp,$(COMP)),openmp)
     CXX=clang++
     COMP_CHOICE=openmp
else ifeq ($(findstring openacc,$(COMP)),openacc)
     CXX=pgc++
     COMP_CHOICE=openacc
else ifeq ($(findstring intel,$(COMP)),intel)
     CXX=icpc
     COMP_CHOICE=intel
else ifeq ($(findstring gcc,$(COMP)),gcc)
     CXX=g++
     COMP_CHOICE=gcc
else ifeq ($(findstring cuda,$(COMP)),cuda)
     CXX=nvcc
     COMP_CHOICE=cuda
else ifeq ($(findstring kokkos,$(COMP)),kokkos)
     CXX=nvcc_wrapper
     COMP_CHOICE=kokkos
else
     $(info #### Usage: make COMP={gcc,intel,cuda,openacc,openmp,kokkos} ####)
     $(info ########   Warning: COMP not set, using gcc by default   ########)
endif

CXXFLAGS = -g -O2 -std=c++11

#
# LLVM COMPILER - clang
#
ifeq ($(COMP_CHOICE),openmp)
  CXXFLAGS += -ffast-math -ffp-contract=fast -fstrict-aliasing
  CXXFLAGS += -Wall -Wextra -Wno-unused-variable
  CXXFLAGS += -fopenmp
  # clang-ykt option: -fopenmp-nonaliased-maps
  CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_ROOT}
  LDFLAGS += -lm

#
# PGI COMPILER - pgcc
#
else ifeq ($(COMP_CHOICE),openacc)
  CXXFLAGS += -fast -acc
  CXXFLAGS += -ta=tesla:fastmath
  CXXFLAGS += -ta:tesla:managed
  CXXFLAGS += -Minfo=accel

#
# CUDA COMPILER - nvcc
#
else ifeq ($(COMP_CHOICE),cuda)
  cudaflags:=-arch=sm_70
  CXXFLAGS += -Wno-deprecated-gpu-targets $(cudaflags)
  CXXFLAGS += -I$(CUDA_PATH)/samples/common/inc/

#
# Kokkos COMPILER - nvcc_wrapper
#
else ifeq ($(COMP_CHOICE),kokkos)
  cudaflags:=-arch=sm_70
  CXXFLAGS += -Wno-deprecated-gpu-targets $(cudaflags)
  CXXFLAGS += -I$(CUDA_PATH)/samples/common/inc/

#
# INTEL COMPILER - icc
#
else ifeq ($(COMP_CHOICE),intel)
  CXXFLAGS += -Wall -Wextra -Wno-unused-variable
  CXXFLAGS += -qopenmp
  ifeq ($(CRAY_CPU_TARGET),haswell)
      CXXFLAGS += -xcore-avx2
  else ifeq ($(CRAY_CPU_TARGET),mic-knl)
      CXXFLAGS += -xmic-avx512
  endif
  CXXFLAGS += -qopenmp-offload=host

#
# GNU COMPILER - gcc
#
else ifeq ($(COMP_CHOICE),gcc)
  CXXFLAGS += -Wall -Wextra -Wno-unused-variable
  CXXFLAGS += -fstrict-aliasing
  CXXFLAGS += -fopenmp
  CXXFLAGS += -foffload=disable
endif
