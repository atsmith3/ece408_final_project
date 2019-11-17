#include "baseline.h"

#define BLOCK 16

__global__ void forward_kernel_baseline(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
  const int radius = (K-1)/2;
  const int x_out = blockIdx.x*BLOCK + (threadIdx.x - radius);
  const int y_out = blockIdx.y*BLOCK + (threadIdx.y - radius);

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  if((x_out >= 0 && x_out < W_out) && (y_out >= 0 && y_out < H_out)) {
    if(((threadIdx.x - radius) >= 0 && (threadIdx.x - radius) < BLOCK) &&
       ((threadIdx.y - radius) >= 0 && (threadIdx.y - radius) < BLOCK)) {
      for(int b = 0; b < B; ++b) {
        for(int m = 0; m < M; m++) {
          y4d(b,m,y_out,x_out) = 0;
          for(int c = 0; c < C; c++) {
            for(int p = 0; p < K; p++) {
              for(int q = 0; q < K; q++) {
                y4d(b,m,y_out,x_out) += x4d(b,c,y_out+p,x_out+q) * k4d(m,c,p,q);
              }
            }
          }
        }
      }
    }
  }
#undef y4d
#undef x4d
#undef k4d
}

void forward_baseline(Tensor &y, const Tensor &x, const Tensor &w) {
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
  dim3 gridDim(ceil((float)(W-2*radius)/((float)BLOCK)), ceil((float)(H-2*radius)/((float)BLOCK)), 1);
  dim3 blockDim(block, block, 1);

  // Call the kernel
  forward_kernel_baseline<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  cudaDeviceSynchronize();
}

void generate_random() {
  Tensor w;
  Tensor x;
  Tensor w2;
  Tensor x2;

  w.create(12, 1, 5, 5);
  w2.create(24, 12, 5, 5);
  x.create(1000, 1, 70, 70);
  x2.create(1000, 12, 33, 33);

  for(int i = 0; i < x.size(); i++) {
    x.hptr_[i] = (float)(rand()%10000)/10000.0;
  }
  for(int i = 0; i < w.size(); i++) {
    w.hptr_[i] = (float)(rand()%10000)/10000.0;
  }
  for(int i = 0; i < x2.size(); i++) {
    x2.hptr_[i] = (float)(rand()%10000)/10000.0;
  }
  for(int i = 0; i < w2.size(); i++) {
    w2.hptr_[i] = (float)(rand()%10000)/10000.0;
  }

  w.write("data/w1.raw");
  w2.write("data/w2.raw");
  x.write("data/x1.raw");
  x2.write("data/x2.raw");
}

int create_golden() {
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

  forward_baseline(y1, x1, w1);
  y1.copyToHost();

  forward_baseline(y2, x2, w2);
  y2.copyToHost();

  if(!y1.write("data/y1.raw")) return -1;
  if(!y2.write("data/y2.raw")) return -1;
  return 0;
}
