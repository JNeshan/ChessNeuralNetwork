#include "matriceMath.h"

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


__global__ void AddKernelM(const float* input, const float* bias, float* out, const int n, const int m){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n * m){
    if(input == out){
      out[idx] += bias[idx];
    }
    else{
      out[idx] = bias[idx] + input[idx];
    }
  }
}


MatriceMath::MatriceMath(){
  cudnnStatus_t nnErr = cudnnCreate(&cudnn);
  if(nnErr != CUDNN_STATUS_SUCCESS){
    throw std::runtime_error("Bad cudnn");   
  }
  cublasStatus_t blasErr = cublasCreate_v2(&cublas);  
  if(blasErr != CUBLAS_STATUS_SUCCESS){
    throw std::runtime_error("Bad cublas");
  }
  std::cout<<"Handles created"<<std::endl;
}

void MatriceMath::add(const Tensor& N, const Tensor& B, Tensor& O){
  if(N.size != B.size || N.size != O.size){
    throw std::runtime_error("Different sizes on addition");
  }

  const float* bias = B.gpuData(), *input = N.gpuData();
  float* out = O.gpuData();
  int n = N.dimensions[0], m = N.size / n, thrdCnt = 256;
  dim3 gridDim((N.size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
  AddKernelM<<<gridDim, blockDim>>>(input, bias, out, n, m);
  return;
}

void MatriceMath::multiply(const Tensor& M, const Tensor& N, Tensor& O){

}

void MatriceMath::convolution(const Tensor& M, const Tensor& F, Tensor& O){

}


