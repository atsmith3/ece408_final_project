
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK 16

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct AND fast.
  We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  */

  const int radius = (K-1)/2;
  // Calculate output X and Y for the thread
  const int x_out = blockIdx.x*BLOCK + (threadIdx.x - (K-1)/2);
  const int y_out = blockIdx.y*BLOCK + (threadIdx.y - (K-1)/2);

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Verify Output is Valid
  if((x_out > 0 && x_out < W_out) && (y_out > 0 && y_out < H_out)) {
    // Check if the thread is a halo thread
    if(((threadIdx.x - radius) > 0 && (threadIdx.x - radius) < BLOCK) &&
       ((threadIdx.y - radius) > 0 && (threadIdx.y - radius) < BLOCK)) {
      // Do Convolution:
      for(int b = 0; b < B; ++b) {
        for(int m = 0; m < M; m++) {
          y[b][m][y_out][x_out] = 0;
          for(int c = 0; c < C; c++) {
            for(int p = 0; p < K; p++) {
              for(int q = 0; q < K; q++) {
                y[b][m][y_out][x_out] += x[b][c][y_out + p][x_out + q] * k[m][c][p][q];
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
  CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

  // Extract the tensor dimensions into B,M,C,H,W,K
  const int B = x.shape_[0];
  const int M = y.shape_[1];
  const int C = x.shape_[1];
  const int K = k.shape_[3];
  const int H = x.shape_[2];
  const int W = x.shape_[3];

  const int radius = (K-1)/2;
  const int block = BLOCK + 2*radius;

  // Set the kernel dimensions
  dim3 gridDim(ceil((float)(W-2*radius)/(BLOCK), ceil((float)(H-2*radius)/(BLOCK))), 1);
  dim3 blockDim(block, block, 1);

  // Call the kernel
  forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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
