#ifndef LMP_ARRAYMD_H
#define LMP_ARRAYMD_H

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

template <typename T> class Array1D {
public:
  unsigned n1;
  unsigned size;
  T *dptr;

  inline T &operator()(unsigned i1) { return dptr[i1]; }

  Array1D() {
    n1 = 0;
    size = 0;
    dptr = NULL;
  }
  Array1D(const Array1D &p) {
    n1 = p.n1;
    size = 0;
    dptr = p.dptr;
  }
  Array1D(int in1) {
    n1 = in1;
    size = n1;
    dptr = (T *)malloc(size * sizeof(T));
  }
  ~Array1D() {
    if (size && dptr)
      free(dptr);
  }

  unsigned getSizeInBytes() { return size * sizeof(T); } // NB: in bytes
};

template <typename T> class Array2D {
public:
  unsigned n1, n2, b1;
  unsigned size;
  T *dptr;

  inline T &operator()(unsigned i1, unsigned i2) {
    return dptr[i2 + (n2 * i1)];
  }

  Array2D() {
    n1 = n2 = 0;
    size = 0;
    dptr = NULL;
  }
  Array2D(const Array2D &p) {
    n1 = p.n1;
    n2 = p.n2;
    size = 0;
    dptr = p.dptr;
  }
  Array2D(int in1, int in2) {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = (T *)malloc(size * sizeof(T));
  }
  ~Array2D() {
    if (size && dptr)
      free(dptr);
  }

  void resize(unsigned in1, unsigned in2) {
    if (size && dptr)
      free(dptr);
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = (T *)malloc(size * sizeof(T));
  }
  void rebound(unsigned in1, unsigned in2) {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = NULL;
  }

  void setBase(unsigned i1, unsigned i2) { b1 = i1 * n2 + i2; }
  inline T &operator()(unsigned i2) { return dptr[b1 + i2]; }

  unsigned getSizeInBytes() { return size * sizeof(T); } // NB: in bytes
};

template <typename T> class Array3D {
private:
  unsigned f2, f1, b1, b2;

public:
  unsigned n1, n2, n3;
  unsigned size;
  T *dptr;
  inline T &operator()(unsigned i1, unsigned i2, unsigned i3) {
    return dptr[i3 + i2 * f2 + i1 * f1];
  }

  Array3D() {
    n1 = n2 = n3 = 0;
    size = 0;
    dptr = NULL;
  }
  Array3D(const Array3D &p) {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    size = 0;
    dptr = p.dptr;
    f2 = n3;
    f1 = f2 * n2;
  }
  Array3D(unsigned in1, unsigned in2, unsigned in3) {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    dptr = (T *)malloc(size * sizeof(T));
    f2 = n3;
    f1 = f2 * n2;
  }

  ~Array3D() {
    if (size && dptr)
      free(dptr);
  }

  unsigned getSizeInBytes() { return size * sizeof(T); } // NB: in bytes
};

#endif
