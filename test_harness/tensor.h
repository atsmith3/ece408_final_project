#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

class Tensor {
public:
  int shape_[4];
  float* hptr_;
  float* dptr_;

  Tensor();
  ~Tensor();
  bool read(char* fname);
  bool write(char* fname);
  bool create(int a, int b, int c, int d);
  int size();
  void copyToDevice();
  void copyToHost();
};

#endif
