
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float c_kernel[24*12*5*5];

#define BLOCK 16
#define COARSENING_FACTOR 2

__global__ void forward_kernel(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K) {
  __shared__ float in_tile[COARSENING_FACTOR*BLOCK][COARSENING_FACTOR*BLOCK];
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const int out_block = COARSENING_FACTOR*(BLOCK - K + 1);
  const int channel_blocks = ceil(((float)H_out)/((float)out_block));
  const int tx = blockIdx.x*out_block + COARSENING_FACTOR*threadIdx.x;

  const int tz = blockIdx.z*blockDim.z + threadIdx.z;

  const int in_x = tx;
  const int in_y = (blockIdx.y % channel_blocks) * out_block + COARSENING_FACTOR*threadIdx.y;
  const int in_c = blockIdx.y / channel_blocks;
  const int in_b = tz;

  const int out_x = in_x;
  const int out_y = in_y;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) c_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1)* (K) + i0]

  float temp[COARSENING_FACTOR][COARSENING_FACTOR];
  #pragma unroll
  for(int tc_y = 0; tc_y < COARSENING_FACTOR; tc_y++) {
    #pragma unroll
    for(int tc_x = 0; tc_x < COARSENING_FACTOR; tc_x++) {
      if(in_x + tc_x < W && in_y + tc_y < H) {
        in_tile[COARSENING_FACTOR*threadIdx.y+tc_y][COARSENING_FACTOR*threadIdx.x+tc_x] = x4d(in_b, in_c, (in_y + tc_y), (in_x + tc_x));
      }
    }
  }
  __syncthreads();
  if(COARSENING_FACTOR*threadIdx.x < out_block && COARSENING_FACTOR*threadIdx.y < out_block && out_x < W_out && out_y < H_out) {
    for(int m = 0; m < M; m++) {
      #pragma unroll
      for(int tc_y = 0; tc_y < COARSENING_FACTOR; tc_y++) {
        #pragma unroll
        for(int tc_x = 0; tc_x < COARSENING_FACTOR; tc_x++) {
          temp[tc_y][tc_x] = 0.0;
        }
      }
      for(int p = 0; p < K; p++) {
        for(int q = 0; q < K; q++) {
          #pragma unroll
          for(int tc_y = 0; tc_y < COARSENING_FACTOR; tc_y++) {
            #pragma unroll
            for(int tc_x = 0; tc_x < COARSENING_FACTOR; tc_x++) {
              temp[tc_y][tc_x] += in_tile[COARSENING_FACTOR*threadIdx.y + tc_y + p][COARSENING_FACTOR*threadIdx.x + tc_x + q] * k4d(m, in_c, p, q);
            }
          }
        }
      }
      #pragma unroll
      for(int tc_y = 0; tc_y < COARSENING_FACTOR; tc_y++) {
        #pragma unroll
        for(int tc_x = 0; tc_x < COARSENING_FACTOR; tc_x++) {
          atomicAdd(&y4d(in_b, m, (out_y + tc_y), (out_x + tc_x)), temp[tc_y][tc_x]);
        }
      }
    }
  }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void generate_unrolled_kernel(float* __restrict__ k, float* __restrict__ k_unrolled, const int M, const int C, const int K) {
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

#define MM_TILE 8
#define MM_CF 4     // Coarsening Factor
#define MM_CF_Y 4     // Coarsening Factor

__global__ void matrixMultiplyShared(float* __restrict__ in, float* __restrict__ out, float* __restrict__ kernel,
                                     int numInRows, int numInColumns,
                                     int numOutRows, int numOutColumns,
                                     int numKernelRows, int numKernelColumns,
                                     int B, int M, int C, int H, int W, int K) {
  unsigned int H_out = H - K + 1;
  unsigned int W_out = W - K + 1;
#define x4d(i3, i2, i1, i0) in[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    
  __shared__ float subTileKernel[MM_CF*MM_TILE][MM_CF*MM_TILE];
  __shared__ float subTileIn[MM_CF*MM_TILE][MM_CF*MM_TILE];
  
  // Get thread Infos
  unsigned int bx = blockIdx.x*MM_CF;
  unsigned int by = blockIdx.y*MM_CF;
  unsigned int tx = threadIdx.x*MM_CF;
  unsigned int ty = threadIdx.y*MM_CF;
  unsigned int batch_i = threadIdx.z + blockDim.z*blockIdx.z;
  
  // Identify element of C being computed
  unsigned int row = by*MM_TILE + ty;
  unsigned int col = bx*MM_TILE + tx;
  
  // Initialize partial sum to 0
  float partialOut[MM_CF][MM_CF];
  #pragma unroll
  for(int tc_y = 0; tc_y < MM_CF; tc_y++) {
    #pragma unroll
    for(int tc_x = 0; tc_x < MM_CF; tc_x++) {
      partialOut[tc_y][tc_x] = 0.0;
    }
  }
  
  // Loop over the tiles.
  if(batch_i < B) {
    for(int i = 0; i < ceil((float)numKernelColumns/(MM_CF*MM_TILE)); i++) {
      #pragma unroll
      for(int tc_y = 0; tc_y < MM_CF; tc_y++) {
        #pragma unroll
        for(int tc_x = 0; tc_x < MM_CF; tc_x++) {

          // Collaboratively load the tile
          int a_x = i*MM_CF*MM_TILE + tx + tc_x;
          int a_y = row + tc_y;
          int b_x = col + tc_x;
          int b_y = i*MM_CF*MM_TILE + ty + tc_y;

          //unsigned int Y_u = K*K*C;
          //unsigned int X_u = H_out*W_out;
          unsigned int c = b_y/(K*K);
          unsigned int x_k = b_x%W_out;
          unsigned int y_k = b_x/W_out;
          unsigned int x_j = (b_y%(K*K))%K + x_k;
          unsigned int y_j = (b_y%(K*K))/K + y_k;

          if(a_x < numKernelColumns) {
            subTileKernel[ty+tc_y][tx+tc_x] = kernel[a_y*numKernelColumns + a_x];
          }
          else {
            subTileKernel[ty+tc_y][tx+tc_x] = 0;
          }
          if(b_y < numInRows) {
            subTileIn[ty+tc_y][tx+tc_x] = x4d(batch_i,c,y_j,x_j);
          }
          else {
            subTileIn[ty+tc_y][tx+tc_x] = 0;
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for(int tc_y = 0; tc_y < MM_CF; tc_y++) {
        #pragma unroll
        for(int tc_x = 0; tc_x < MM_CF; tc_x++) {
          #pragma unroll 32
          for(int k = 0; k < MM_CF*MM_TILE; k++) {
            partialOut[tc_y][tc_x] += subTileKernel[ty+tc_y][k]*subTileIn[k][tx+tc_x];
          }
        }
      }
      __syncthreads();
    }
    // Before comitting check if its valid
    #pragma unroll
    for(int tc_y = 0; tc_y < MM_CF; tc_y++) {
      #pragma unroll
      for(int tc_x = 0; tc_x < MM_CF; tc_x++) {
        if((row + tc_y) < numOutRows && (col + tc_x) < numOutColumns) {
          unsigned int out_m = row + tc_y;
          unsigned int out_x = (col + tc_x)%W_out;
          unsigned int out_y = (col + tc_x)/W_out;
#define y4d(i3, i2, i1, i0) out[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
          y4d(batch_i,out_m,out_y,out_x) = partialOut[tc_y][tc_x];
#undef y4d
        }
      }
    }
  }
#undef x4d
}

uint64_t op = 1;

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

  if(op == 2) {
    float* w_unrolled;

    MSHADOW_CUDA_CALL(cudaMalloc(&w_unrolled, M*C*K*K*sizeof(float)));

    // Format Inputs:
    dim3 gridDimUK(ceil((float)(K*K*C)/((float)MM_TILE)), ceil((float)(M)/((float)MM_TILE)));
    dim3 blockDimUK(MM_TILE, MM_TILE);
    generate_unrolled_kernel<<<gridDimUK, blockDimUK>>>(w.dptr_, w_unrolled, M, C, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    
    // Mat Mul:
    //cudaMemcpyToSymbol(c_kernel, w.dptr_, M*C*K*K*sizeof(float));
    dim3 gridDimMM(ceil((float)(H_out*W_out)/((float)MM_CF*MM_TILE)), ceil((float)(M)/((float)MM_CF*MM_TILE)), ceil((float)(B)/(float)1));
    dim3 blockDimMM(MM_TILE, MM_TILE, 1);
    matrixMultiplyShared<<<gridDimMM, blockDimMM>>>(x.dptr_, /*y_unrolled*/y.dptr_, w.dptr_, K*K*C, H_out*W_out, M, H_out*W_out, M, K*K*C, B, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    cudaFree(w_unrolled);
    op++;
  }
  
  if(op == 1) {
    const int out_block = COARSENING_FACTOR*(BLOCK - K + 1);

    // Set the kernel dimensions
    cudaMemcpyToSymbol(c_kernel, w.dptr_, M*C*K*K*sizeof(float));
    dim3 blockDim(BLOCK, BLOCK, 1);
    dim3 gridDim(ceil(((float)W_out)/((float)out_block)), ceil(((float)(H_out))/((float)out_block))*C, B);
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    op++;
  }
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
