
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
  int x_i = threadIdx.x + blockDim.x*blockIdx.x;
  int y_i = threadIdx.y + blockDim.y*blockIdx.y;

  int m = y_i;
  int c = x_i/(K*K);
  int x_j = (x_i%(K*K))%K;
  int y_j = (x_i%(K*K))/K;

  if(x_i < K*K*C && y_i < M) {
    ku2d(y_i,x_i) = k4d(m,c,y_j,x_j);
  }
#undef k4d
#undef ku2d
}

__global__ void generate_unrolled(float* x, float* x_unrolled, const int B, const int C, const int H, const int W, const int K) {
  int H_out = H - K + 1;
  int W_out = H - K + 1;
  int b_i = threadIdx.z + blockDim.z*blockIdx.z;
  int x_i = threadIdx.x + blockDim.x*blockIdx.x;
  int y_i = threadIdx.y + blockDim.y*blockIdx.y;
  int Y_u = K*K*C;
  int X_u = H_out*W_out;
  int c = y_i/(K*K);
  int x_k = x_i%W_out;
  int y_k = x_i/W_out;
  int x_j = (y_i%(K*K))%K + x_k;
  int y_j = (y_i%(K*K))/K + y_k;
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define xu3d(i2, i1, i0) x_unrolled[(i2) * (Y_u * X_u) + (i1) * (X_u) + i0]
  if(b_i < B && x_i < X_u && y_i < Y_u) {
    xu3d(b_i,y_i,x_i) = x4d(b_i,c,y_j,x_j);
  }
#undef x4d
#undef xu3d
}

__global__ void generate_rolled(float* y, float* y_unrolled, const int B, const int M, const int H, const int W, const int K) {
  int H_out = H - K + 1;
  int W_out = H - K + 1;
  int b_i = threadIdx.z + blockDim.z*blockIdx.z;
  int x_i = threadIdx.x + blockDim.x*blockIdx.x;
  int y_i = threadIdx.y + blockDim.y*blockIdx.y;
  int Y_u = M;
  int X_u = H_out*W_out;
  int m = y_i;
  int x_j = x_i%W_out;
  int y_j = x_i/W_out;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define yu3d(i2, i1, i0) y_unrolled[(i2) * (Y_u * X_u) + (i1) * (X_u) + i0]
  if(b_i < B && x_i < X_u && y_i < Y_u) {
    y4d(b_i,m,y_j,x_j) = yu3d(b_i,y_i,x_i);
  }
#undef y4d
#undef yu3d
}

#define BLOCK 8
#define C_BLOCK 7


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
#define k4d(i3, i2, i1, i0) c_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
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

#define MM_TILE 32

__global__ void matrixMultiplyShared(float *in, float *out, float *kernel,
                                     int numInRows, int numInColumns,
                                     int numOutRows, int numOutColumns,
                                     int numKernelRows, int numKernelColumns,
                                     int B) {
    
  __shared__ float subTileKernel[MM_TILE][MM_TILE];
  __shared__ float subTileIn[MM_TILE][MM_TILE];
  
  // Get thread Infos
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int batch_i = threadIdx.z + blockDim.z+blockIdx.z;
  
  // Identify element of C being computed
  int row = by*MM_TILE + ty;
  int col = bx*MM_TILE + tx;
  
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
      
      if(a_x < numKernelColumns) {
        subTileKernel[ty][tx] = kernel[a_y*numKernelColumns + a_x];
      }
      if(b_y < numInRows) {
        subTileIn[ty][tx] = in[batch_i*(numInColumns*numInRows) + b_y*numInColumns + b_x];
      }
      __syncthreads();
      for(int k = 0; k < MM_TILE; k++) {
        if(i*MM_TILE+k < numKernelColumns && row < numOutRows && col < numOutColumns)
        partialOut += subTileKernel[ty][k]*subTileIn[k][tx];
      }
      __syncthreads();
    }
    // Before comitting check if its valid
    if(row < numOutRows && col < numOutColumns) {
      out[batch_i*(numOutColumns*numOutRows)+ row*numOutColumns + col] = partialOut;
    }
  }
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

  float* x_unrolled;
  float* y_unrolled;
  float* w_unrolled;

  cudaMalloc(&x_unrolled, B*C*H*W*sizeof(float));
  cudaMalloc(&y_unrolled, B*M*H_out*W_out*sizeof(float));
  cudaMalloc(&w_unrolled, M*C*K*K*sizeof(float));

  // Format Inputs:
  dim3 gridDimUK(ceil((float)(K*K*C)/((float)MM_TILE)), ceil((float)(M)/((float)MM_TILE)));
  dim3 blockDimUK(MM_TILE, MM_TILE);
  generate_unrolled_kernel<<<gridDimUK, blockDimUK>>>(w.dptr_, w_unrolled, M, C, K);
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
  
  dim3 gridDimX(ceil((float)(H_out*W_out)/((float)MM_TILE)), ceil((float)(K*K*C)/((float)MM_TILE)), ceil((float)(B)/(float)1));
  dim3 blockDimX(MM_TILE, MM_TILE, 1);
  generate_unrolled<<<gridDimX, blockDimX>>>(x.dptr_, x_unrolled, B, C, H, W, K);
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

  // Mat Mul:
  dim3 gridDimMM(ceil((float)(H_out*W_out)/((float)MM_TILE)), ceil((float)(M)/((float)MM_TILE)), ceil((float)(B)/(float)1));
  dim3 blockDimMM(MM_TILE, MM_TILE, 1);
  matrixMultiplyShared<<<gridDimMM, blockDimMM>>>(x_unrolled, y_unrolled, w_unrolled, K*K*C, H_out*W_out, M, H_out*W_out, M, K*K*C, B);
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());


  // Format Output:
  dim3 gridDimY(ceil((float)(H_out*W_out)/((float)MM_TILE)), ceil((float)(M)/((float)MM_TILE)), ceil((float)(B)/(float)1));
  dim3 blockDimY(MM_TILE, MM_TILE, 1);
  generate_rolled<<<gridDimY, blockDimY>>>(y.dptr_, y_unrolled, B, M, H, W, K);

  // Set the kernel dimensions
  //cudaMemcpyToSymbol(kernel, w_unrolled, M*C*K*K*sizeof(float));
  //cudaMemset((void*)y.dptr_, 0, B*M*H_out*W_out*sizeof(float));
  //dim3 gridDim(ceil((float)(W-K+1)/((float)BLOCK)), ceil((float)(H-K+1)/((float)BLOCK)), ceil((float)(C)/(float)C_BLOCK));
  //dim3 blockDim(block, block, C_BLOCK);

  // Call the kernel
  //forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
  //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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
