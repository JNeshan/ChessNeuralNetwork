#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas.h>
#include "header/optimizer.h"

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCuda(cublasStatus_t err){
  if(err != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "cuBLAS Error in %s at line %d: %s\n", __FILE__, __LINE__, cublasGetStatusString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCuda(cudnnStatus_t err){
  if(err != CUDNN_STATUS_SUCCESS){
    fprintf(stderr, "CUDNN Error in %s at line %d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(err));
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
  const int m = in.dimensions[0], n =  in.size / m;
  int thCount = 256;
  while(thCount < m){
    thCount *= 2;
  }
  dim3 gridDim(m);
  dim3 blockDim(thCount);
  GradDescentKernel<<<gridDim, blockDim>>>(grad.gpuData(), in.gpuData(), lR, m, n);
}