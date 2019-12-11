
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float c_kernel[24*12*5*5];

__global__ void generate_unrolled_kernel(float* k, float* k_unrolled, const int M, const int C, const int K) {
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define ku2d(i1, i0) k_unrolled[(i1) * (K*K*C) + i0]
  unsigned int x_i = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int y_i = threadIdx.y + blockDim.y*blockIdx.y;

  unsigned int m = y_i;
  unsigned int c = x_i/(K*K);
  unsigned int x_j = (x_i%(K*K))%K;
  unsigned int y_j = (x_i%(K*K))/K;

  if(x_i < K*K*C && y_i < M) {
    ku2d(y_i,x_i) = k4d(m,c,y_j,x_j);
  }
#undef k4d
#undef ku2d
}

#define MM_TILE 32

__global__ void matrixMultiplyShared(float *in, float *out, float *kernel,
                                     int numInRows, int numInColumns,
                                     int numOutRows, int numOutColumns,
                                     int numKernelRows, int numKernelColumns,
                                     int B, int M, int C, int H, int W, int K) {
  unsigned int H_out = H - K + 1;
  unsigned int W_out = W - K + 1;
#define x4d(i3, i2, i1, i0) in[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    
  __shared__ float subTileKernel[MM_TILE][MM_TILE];
  __shared__ float subTileIn[MM_TILE][MM_TILE];
  
  // Get thread Infos
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int batch_i = threadIdx.z + blockDim.z*blockIdx.z;
  
  // Identify element of C being computed
  unsigned int row = by*MM_TILE + ty;
  unsigned int col = bx*MM_TILE + tx;
  
  // Initialize partial sum to 0
  float partialOut = 0.0;
  
  // Loop over the tiles.
  if(batch_i < B) {
    for(int i = 0; i < ceil((float)numKernelColumns/MM_TILE); i++) {
      // Collaboratively load the tile
      int a_x = i*MM_TILE + tx;
      int a_y = row;
      int b_x = col;
      int b_y = i*MM_TILE + ty;

      //unsigned int Y_u = K*K*C;
      //unsigned int X_u = H_out*W_out;
      unsigned int c = b_y/(K*K);
      unsigned int x_k = b_x%W_out;
      unsigned int y_k = b_x/W_out;
      unsigned int x_j = (b_y%(K*K))%K + x_k;
      unsigned int y_j = (b_y%(K*K))/K + y_k;
      
      if(a_x < numKernelColumns) {
        subTileKernel[ty][tx] = kernel[a_y*numKernelColumns + a_x];
      }
      else {
        subTileKernel[ty][tx] = 0;
      }
      if(b_y < numInRows) {
        subTileIn[ty][tx] = x4d(batch_i,c,y_j,x_j);
      }
      else {
        subTileIn[ty][tx] = 0;
      }
      __syncthreads();
      #pragma unroll 32
      for(int k = 0; k < MM_TILE; k++) {
        partialOut += subTileKernel[ty][k]*subTileIn[k][tx];
      }
      __syncthreads();
    }
    // Before comitting check if its valid
    if(row < numOutRows && col < numOutColumns) {
      unsigned int out_m = row;
      unsigned int out_x = col%W_out;
      unsigned int out_y = col/W_out;
#define y4d(i3, i2, i1, i0) out[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
      y4d(batch_i,out_m,out_y,out_x) = partialOut;
#undef y4d
    }
  }
#undef x4d
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

  //const int block = BLOCK + K - 1;

  float* w_unrolled;

  MSHADOW_CUDA_CALL(cudaMalloc(&w_unrolled, M*C*K*K*sizeof(float)));

  // Format Inputs:
  dim3 gridDimUK(ceil((float)(K*K*C)/((float)MM_TILE)), ceil((float)(M)/((float)MM_TILE)));
  dim3 blockDimUK(MM_TILE, MM_TILE);
  generate_unrolled_kernel<<<gridDimUK, blockDimUK>>>(w.dptr_, w_unrolled, M, C, K);
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
  
  // Mat Mul:
  dim3 gridDimMM(ceil((float)(H_out*W_out)/((float)MM_TILE)), ceil((float)(M)/((float)MM_TILE)), ceil((float)(B)/(float)1));
  dim3 blockDimMM(MM_TILE, MM_TILE, 1);
  matrixMultiplyShared<<<gridDimMM, blockDimMM>>>(x.dptr_, /*y_unrolled*/y.dptr_, w_unrolled, K*K*C, H_out*W_out, M, H_out*W_out, M, K*K*C, B, M, C, H, W, K);
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

  // Set the kernel dimensions
  //cudaMemcpyToSymbol(kernel, w_unrolled, M*C*K*K*sizeof(float));
  //cudaMemset((void*)y.dptr_, 0, B*M*H_out*W_out*sizeof(float));
  //dim3 gridDim(ceil((float)(W-K+1)/((float)BLOCK)), ceil((float)(H-K+1)/((float)BLOCK)), ceil((float)(C)/(float)C_BLOCK));
  //dim3 blockDim(block, block, C_BLOCK);

  // Call the kernel
  //forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
  //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
  cudaFree(w_unrolled);

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
