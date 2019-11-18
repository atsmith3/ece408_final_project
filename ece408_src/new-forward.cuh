
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

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
  __shared__ float in_tile[7][12][12];

  // Calculate output X and Y for the thread
  const int tx = blockIdx.x*BLOCK + threadIdx.x;
  const int ty = blockIdx.y*BLOCK + threadIdx.y;
  const int tc = blockIdx.z*blockDim.z + threadIdx.z;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  float temp = 0;

  // Verify Output is Valid
  for(int b = 0; b < B; ++b) {
    // Load Input Tile:
    if(tx < W && ty < H && tc < C) {
      in_tile[threadIdx.z][threadIdx.y][threadIdx.x] = x4d(b,tc,ty,tx);
    }
    __syncthreads();
    for(int m = 0; m < M; m++) {
      if(tx < W_out && ty < H_out && tc < C) {
        if(threadIdx.x < BLOCK && threadIdx.y < BLOCK) {
          temp = 0;
#pragma unroll
          for(int p = 0; p < K; p++) {
#pragma unroll
            for(int q = 0; q < K; q++) {
              //y[b][m][y_out][x_out] += x[b][c][y_out + p][x_out + q] * k[m][c][p][q];
              //temp += x4d(b,tc,ty+p,tx+q) * k4d(m,tc,p,q);
              temp += in_tile[threadIdx.z][threadIdx.y+p][threadIdx.x+q] * k4d(m,tc,p,q);
            }
          }
          atomicAdd(&(y4d(b,m,ty,tx)), temp);
        }
      }
    }
    __syncthreads();
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
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const int block = BLOCK + K - 1;

  // Set the kernel dimensions
  cudaMemcpyToSymbol(kernel, w.dptr_, M*C*K*K*sizeof(float));
  //cudaMemset((void*)y.dptr_, 0, B*M*H_out*W_out*sizeof(float));
  dim3 gridDim(ceil((float)(W-K+1)/((float)BLOCK)), ceil((float)(H-K+1)/((float)BLOCK)), ceil((float)(C)/(float)C_BLOCK));
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
