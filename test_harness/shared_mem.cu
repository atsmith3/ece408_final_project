#include <chrono>

#include "shared_mem.h"

#define BLOCK 8
#define C_BLOCK 7

__constant__ float kernel[24*12*5*5];

__global__ void forward_kernel_shared_mem(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
  __shared__ float in_tile[12][12][12];

  const int radius = (K-1)/2;
  const int x_out = blockIdx.x*BLOCK + (threadIdx.x - radius);
  const int y_out = blockIdx.y*BLOCK + (threadIdx.y - radius);
  const int c = blockIdx.z*blockDim.z + threadIdx.z;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  const int tx = threadIdx.x - radius;
  const int ty = threadIdx.y - radius;

  const int ix = threadIdx.x + blockIdx.x*BLOCK;
  const int iy = threadIdx.y + blockIdx.y*BLOCK;
  const int ic = threadIdx.z + blockIdx.z*blockDim.z;

  float temp = 0;

  for(int b = 0; b < B; ++b) {
    if(ix < W && iy < H && ic < C) {
      in_tile[threadIdx.z][threadIdx.y][threadIdx.x] = x4d(b,ic,iy,ix);
    }
    __syncthreads();
    for(int m = 0; m < M; m++) {
      if((x_out >= 0 && x_out < W_out) && (y_out >= 0 && y_out < H_out)) {
        if(((tx) >= 0 && (tx) < BLOCK) && ((ty) >= 0 && (ty) < BLOCK)) {
          temp = 0;
          if(c < C) {
            for(int p = 0; p < K; p++) {
              for(int q = 0; q < K; q++) {
                temp += in_tile[threadIdx.z][ty+p][tx+q] * k4d(m,c,p,q);
              }
            }
          }
          atomicAdd(&(y4d(b,m,y_out,x_out)), temp);
        }
      }
    }
    __syncthreads();
  }
#undef y4d
#undef x4d
#undef k4d
}

void forward_shared_mem(Tensor &y, const Tensor &x, const Tensor &w) {
  // Extract the tensor dimensions into B,M,C,H,W,K
  const int B = x.shape_[0];
  const int M = y.shape_[1];
  const int C = x.shape_[1];
  const int K = w.shape_[3];
  const int H = x.shape_[2];
  const int W = x.shape_[3];

  const int radius = (K-1)/2;
  const int block = BLOCK + 2*radius;
  
  // Set the kernel dimensions
  cudaMemcpyToSymbol(kernel, w.dptr_, M*C*K*K*sizeof(float));
  dim3 gridDim(ceil((float)(W-2*radius)/((float)BLOCK)), ceil((float)(H-2*radius)/((float)BLOCK)), ceil((float)(C)/(float)C_BLOCK));
  dim3 blockDim(block, block, C_BLOCK);

  // Call the kernel
  forward_kernel_shared_mem<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

  cudaDeviceSynchronize();
}

int run_shared_mem() {
  Tensor w1;
  Tensor x1;
  Tensor y1;
  Tensor w2;
  Tensor x2;
  Tensor y2;

  //generate_random();

  if(!y1.create(1000, 12, 66, 66)) return -1;
  if(!y2.create(1000, 24, 29, 29)) return -1;

  if(!x1.read("data/x1.raw")) return -1;
  if(!w1.read("data/w1.raw")) return -1;
  if(!x2.read("data/x2.raw")) return -1;
  if(!w2.read("data/w2.raw")) return -1;

  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
  std::chrono::duration<double> time_span;
  t1 = std::chrono::high_resolution_clock::now();
  forward_shared_mem(y1, x1, w1);
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "Op1 Shared Mem: " << time_span.count() << " seconds\n";
  y1.copyToHost();

  t1 = std::chrono::high_resolution_clock::now();
  forward_shared_mem(y2, x2, w2);
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "Op2 Shared Mem: " << time_span.count() << " seconds\n";
  y2.copyToHost();

  if(!y1.write("data/y1_sm.raw")) return -1;
  if(!y2.write("data/y2_sm.raw")) return -1;
  return 0;
}
