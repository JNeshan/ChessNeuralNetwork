#include "header/denseLayer.h"
#include <cuda_runtime.h>
#include <cublas.h>
#include <cudnn.h>
#include <stdexcept>

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

struct CudaMembers{
  cublasHandle_t handle;
  cudnnHandle_t nHandle;
  cudnnTensorDescriptor_t outD, biasD;

  CudaMembers(){
    cublasCreate_v2(&handle);
    cudnnCreate(&nHandle);
    cudnnCreateTensorDescriptor(&outD);
    cudnnCreateTensorDescriptor(&biasD);
  }

  ~CudaMembers(){}
};

DenseLayer::DenseLayer(const int f, const int n) : weight({f, n}, TensorLocation::GPU), bias({1, n}, TensorLocation::GPU){
  CudaM = new CudaMembers();
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n));
}


Tensor DenseLayer::forward(const Tensor& T){
  if(T.dimensions[3] != weight.dimensions[2]){
    throw("Weight and input tensor incorrect dimensions for multiplication");
  }

  Tensor output({T.dimensions[0], weight.dimensions[1]}, TensorLocation::GPU);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->outD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, output.dimensions[2], output.dimensions[3]));
  TryCuda(cublasSgemm_v2(CudaM->handle, CUBLAS_OP_N, CUBLAS_OP_N, weight.dimensions[3], T.dimensions[2], 
                        T.dimensions[3], &mx, weight.gpuData(), weight.dimensions[3], T.gpuData(), 
                        T.dimensions[3], &mn, output.gpuData(), output.dimensions[3]));
  TryCuda(cudnnAddTensor(CudaM->nHandle, &mx, CudaM->outD, output.gpuData(), &mx, CudaM->biasD, bias.gpuData()));
  return output;
}

DenseLayer::~DenseLayer(){
  delete CudaM;
}