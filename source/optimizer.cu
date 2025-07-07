#include <cuda_runtime.h>
#include "header/optimizer.h"

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__global__ void GradDescentKernel(const float* grad, float* in, const float lR, const int m, const int n){
  const int colIdx = blockIdx.x;
  int thId = threadIdx.x;
  for(int row = thId; row < m; row++){
    in[row * n + colIdx] -= (lR * (grad[row * n + colIdx]));
  }
  __syncthreads();
}

struct CudaMembers{
  CudaMembers(){

  }

  ~CudaMembers(){

  };

  void resetTemp(){

  }
};

Optimizer::Optimizer(const float rate) : lR(rate){}

void Optimizer::optimize(const Tensor& in, const Tensor& grad){
  const int m = in.dimensions[0], n =  in.size / m; //row and column dimensions
  int thCount = 256; //number of threads per thread block
  while(thCount < m){ //number of threads must be at least the number of elements per row
    thCount *= 2;
  }
  dim3 gridDim(m); //variables used by the cuda kernel for thread blocks dimensions and counts
  dim3 blockDim(thCount);
  GradDescentKernel<<<gridDim, blockDim>>>(grad.gpuData(), in.gpuData(), lR, m, n); //calls the kernel to perform the optimization operation with tensor and its gradient
  return;
}