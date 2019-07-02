/*
 * Memory class to separate memory allocation from multi-dimensional array
 * access
 */

#ifndef _MEMCLASS_H
#define _MEMCLASS_H
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

template <typename T> class allocMem {
private:
public:
  T *ptr;
  __device__ __host__ allocMem() { ptr = NULL; }
  __device__ __host__ ~allocMem() { ; }
};

#endif
