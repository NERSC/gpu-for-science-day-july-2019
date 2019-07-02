
#ifndef _ARRAYMDGPU_H
#define _ARRAYMDGPU_H

#include "memoryClass.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#define CUDA_HOSTDEV __host__ __device__

// Copy to Device
template <typename T> inline void copyToDevice(T *dest, T *src, size_t size) {
  checkCudaErrors(
      cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice));
}

// Copy to Host
template <typename T> inline void copyToHost(T *dest, T *src, size_t size) {
  checkCudaErrors(
      cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

// for 1d array on device
template <typename T> class device_Array1D {
public:
  size_t n1 = 0;
  size_t size = 0;
  T *dptr;
  allocMem<T> d_dptr;

  CUDA_HOSTDEV inline T &operator()(size_t i1) { return dptr[i1]; }

  CUDA_HOSTDEV device_Array1D() {
    n1 = 0;
    size = 0;
    dptr = NULL;
  }
  CUDA_HOSTDEV device_Array1D(const device_Array1D &p) {
    n1 = p.n1;
    dptr = p.dptr;
  }

  CUDA_HOSTDEV device_Array1D(size_t in1) {
    size = in1;
    n1 = in1;
    checkCudaErrors(cudaMalloc((void **)&d_dptr.ptr, size * sizeof(T)));
    dptr = d_dptr.ptr;
  }

  ~device_Array1D() {
    if (size && d_dptr.ptr)
      checkCudaErrors(cudaFree(dptr));
  }

  unsigned getSizeInBytes() { return size * sizeof(T); } // NB: in bytes
};

// 2D array on Device
template <typename T> class device_Array2D {
public:
  size_t n1 = 0, n2 = 0;
  size_t size = 0;
  T *dptr;
  allocMem<T> d_dptr;

  CUDA_HOSTDEV inline T &operator()(size_t i1, size_t i2) {
    return dptr[i2 + (n2 * i1)];
  }

  CUDA_HOSTDEV device_Array2D() {
    n1 = n2 = 0;
    size = 0;
    dptr = NULL;
  }
  CUDA_HOSTDEV device_Array2D(const device_Array2D &p) {
    n1 = p.n1;
    n2 = p.n2;
    size = 0;
    dptr = p.dptr;
  }
  CUDA_HOSTDEV device_Array2D(size_t in1, size_t in2) {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    checkCudaErrors(cudaMalloc((void **)&d_dptr.ptr, size * sizeof(T)));
    dptr = d_dptr.ptr;
  }

  ~device_Array2D() {
    if (size && dptr)
      checkCudaErrors(cudaFree(dptr));
  }

  unsigned getSizeInBytes() { return size * sizeof(T); } // NB: in bytes
};

// 3D array on Device
template <typename T> class device_Array3D {
private:
  unsigned f2, f1, b1, b2;

public:
  size_t n1 = 0, n2 = 0, n3 = 0;
  size_t size = 0;
  T *dptr;
  allocMem<T> d_dptr;

  CUDA_HOSTDEV inline T &operator()(size_t i1, size_t i2, size_t i3) {
    return dptr[i3 + i2 * n3 + i1 * n2 * n3];
  }

  CUDA_HOSTDEV device_Array3D() {
    n1 = n2 = n3 = 0;
    size = 0;
    dptr = NULL;
  }
  CUDA_HOSTDEV device_Array3D(const device_Array3D &p) {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    size = 0;
    dptr = p.dptr;
  }

  CUDA_HOSTDEV device_Array3D(size_t in1, size_t in2, size_t in3) {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    f2 = n3;
    f1 = f2 * n2;
    size = n1 * n2 * n3;
    checkCudaErrors(cudaMalloc((void **)&d_dptr.ptr, size * sizeof(T)));
    dptr = d_dptr.ptr;
  }

  ~device_Array3D() {
    if (size && dptr)
      checkCudaErrors(cudaFree(dptr));
  }

  unsigned getSizeInBytes() { return size * sizeof(T); } // NB: in bytes
};
#endif
