#include "tensor.h"

Tensor::Tensor() {
  shape_[0] = 0;
  shape_[1] = 0;
  shape_[2] = 0;
  shape_[3] = 0;
  hptr_ = NULL;
  dptr_ = NULL;
}

Tensor::~Tensor() {
  shape_[0] = 0;
  shape_[1] = 0;
  shape_[2] = 0;
  shape_[3] = 0;
  if(hptr_ != NULL) {
    free(hptr_);
    hptr_ = NULL;
  }
  if(dptr_ != NULL) {
    cudaFree(dptr_);
    dptr_ = NULL;
  }
} 

bool Tensor::read(char* fname) {
  FILE* f = fopen(fname, "r");
  if(f == NULL) {
    std::cerr << "ERROR: Could not open file: " << std::string(fname) << "\n";
    return false;
  }
  fscanf(f, "%d\n%d\n%d\n%d\n", &shape_[0], &shape_[1], &shape_[2], &shape_[3]); 
  if(!(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0 && shape_[3] > 0)) {
    std::cerr << "ERROR: Invalid Tensor Dimensions: " << shape_[0] << " " << shape_[1] << " " << shape_[2] << " " << shape_[3] << "\n";
    return false;
  }
  hptr_ = (float*)malloc(shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float));
  if(hptr_ == NULL) {
    std::cerr << "ERROR: Could not allocate space on host\n";
    return false;
  }
  cudaMalloc(&dptr_, shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float));
  if(dptr_ == NULL) {
    std::cerr << "ERROR: Could not allocate space on device\n";
    return false;
  }
  for(int i = 0; i < shape_[0]*shape_[1]*shape_[2]*shape_[3]; i++) {
    fscanf(f, "%f\n", &hptr_[i]);
  }
  fclose(f);
  return true;
}

bool Tensor::write(char* fname) {
  FILE* f = fopen(fname, "w");
  if(f == NULL) {
    std::cerr << "ERROR: Could not open file: " << std::string(fname) << "\n";
    return false;
  }
  fprintf(f, "%d\n%d\n%d\n%d\n", shape_[0], shape_[1], shape_[2], shape_[3]); 
  for(int i = 0; i < shape_[0]*shape_[1]*shape_[2]*shape_[3]; i++) {
    fprintf(f, "%f\n", hptr_[i]);
  }
  fclose(f);
  return true;
}

bool Tensor::create(int a, int b, int c, int d) {
  shape_[0] = a;
  shape_[1] = b;
  shape_[2] = c;
  shape_[3] = d;
  if(!(shape_[0] > 0 && shape_[1] > 0 && shape_[2] > 0 && shape_[3] > 0)) {
    std::cerr << "ERROR: Invalid Tensor Dimensions: " << shape_[0] << " " << shape_[1] << " " << shape_[2] << " " << shape_[3] << "\n";
    return false;
  }
  hptr_ = (float*)malloc(shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float));
  if(hptr_ == NULL) {
    std::cerr << "ERROR: Could not allocate space on host\n";
    return false;
  }
  memset(hptr_, 0, shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float));
  cudaMalloc(&dptr_, shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float));
  if(dptr_ == NULL) {
    std::cerr << "ERROR: Could not allocate space on device\n";
    return false;
  }
  cudaMemset(dptr_, 0, shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float));
  return true;
}

int Tensor::size() {
  return shape_[0]*shape_[1]*shape_[2]*shape_[3];
}

void Tensor::copyToDevice() {
  cudaMemcpy(dptr_, hptr_, shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::copyToHost() {
  cudaMemcpy(hptr_, dptr_, shape_[0]*shape_[1]*shape_[2]*shape_[3]*sizeof(float), cudaMemcpyDeviceToHost);
}
