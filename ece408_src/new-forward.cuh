
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK 32

__constant__ float kernel[24*12*5*5];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
  __shared__ float in_tile[BLOCK][BLOCK];

  // Calculate output X and Y for the thread
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const int out_block = BLOCK - K + 1;
  const int channel_blocks = ceil(((float)H_out)/((float)out_block));
  const int tx = blockIdx.x*out_block + threadIdx.x;
//  const int ty = blockIdx.y*out_block + threadIdx.y;
  const int tz = blockIdx.z*blockDim.z + threadIdx.z;

  const int in_x = tx;
  const int in_y = (blockIdx.y % channel_blocks) * out_block + threadIdx.y;
  const int in_c = blockIdx.y / channel_blocks;
  const int in_b = tz;

  const int out_x = in_x;
  const int out_y = in_y;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  float temp = 0;

  if(in_x < W && in_y < H) {
    in_tile[threadIdx.y][threadIdx.x] = x4d(in_b, in_c, in_y, in_x);
  }
  __syncthreads();
  if(threadIdx.x < out_block && threadIdx.y < out_block && out_x < W_out && out_y < H_out) {
    for(int m = 0; m < M; m++) {
      int out_m = (m + in_c) % M;
      temp = 0;
      for(int p = 0; p < K; p++) {
        for(int q = 0; q < K; q++) {
          temp += in_tile[threadIdx.y + p][threadIdx.x + q] * k4d(out_m, in_c, p, q);
	}
      }
      atomicAdd(&y4d(in_b, out_m, out_y, out_x), temp);
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
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const int out_block = BLOCK - K + 1;

  // Set the kernel dimensions
  cudaMemcpyToSymbol(kernel, w.dptr_, M*C*K*K*sizeof(float));
  //cudaMemset((void*)y.dptr_, 0, B*M*H_out*W_out*sizeof(float));
  dim3 blockDim(BLOCK, BLOCK, 1);
  dim3 gridDim(ceil(((float)W_out)/((float)out_block)), ceil(((float)(H_out))/((float)out_block))*C, B);

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
