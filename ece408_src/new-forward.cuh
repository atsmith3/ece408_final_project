
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK 8
#define C_BLOCK 7

__constant__ float kernel[24*12*5*5];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct AND fast.
  We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  */
  __shared__ float in_tile[12][12][12];

  const int radius = (K-1)/2;
  // Calculate output X and Y for the thread
  const int x_out = blockIdx.x*BLOCK + (threadIdx.x - (K-1)/2);
  const int y_out = blockIdx.y*BLOCK + (threadIdx.y - (K-1)/2);
  const int c = blockIdx.z*blockDim.z + threadIdx.z;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  const int tx = threadIdx.x - radius;
  const int ty = threadIdx.y - radius;

  const int ix = threadIdx.x + blockIdx.x*BLOCK;
  const int iy = threadIdx.y + blockIdx.y*BLOCK;
  const int ic = threadIdx.z + blockIdx.z*blockDim.z;

  float temp = 0;

  // Verify Output is Valid
  for(int b = 0; b < B; ++b) {
    // Load Input Tile:
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
                //y[b][m][y_out][x_out] += x[b][c][y_out + p][x_out + q] * k[m][c][p][q];
                //temp += x4d(b,c,y_out+p,x_out+q) * k4d(m,c,p,q);
                temp += in_tile[threadIdx.z][ty+p][tx+q] * k4d(m,c,p,q);
              }
            }
          }
          atomicAdd(&(y4d(b,m,y_out,x_out)), temp);
        }
      }
    }
  }


#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

  // Use mxnet's CHECK_EQ to do assertions.
  // Remove this assertion when you do your implementation!
  // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

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
  //cudaMemset((void*)y.dptr_, 0, B*M*H*W*sizeof(float));
  dim3 gridDim(ceil((float)(W-2*radius)/((float)BLOCK)), ceil((float)(H-2*radius)/((float)BLOCK)), ceil((float)(C)/(float)C_BLOCK));
  dim3 blockDim(block, block, C_BLOCK);

  // Call the kernel
  //forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
  forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
