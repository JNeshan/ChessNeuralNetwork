#include "header/reluLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>


__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
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
  cudnnHandle_t handle;
  cudnnActivationDescriptor_t reLU;
  cudnnTensorDescriptor_t inputD;
  cudnnTensorDescriptor_t outputD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
    TryCuda(cudnnCreateActivationDescriptor(&reLU));
  }
  ~CudaMembers(){
  }
};

ReLULayer::ReLULayer(){
  CudaM = new CudaMembers();
}

Tensor ReLULayer::forward(const Tensor& T){
  Tensor output(T.dimensions, TensorLocation::GPU);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], T.dimensions[2], T.dimensions[3]));
  TryCuda(cudnnSetActivationDescriptor(CudaM->reLU, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 2));
  TryCuda(cudnnActivationForward(CudaM->handle, CudaM->reLU, &alpha, CudaM->inputD, T.gpuData(), &beta, CudaM->outputD, output.gpuData()));
  return output;
}

void ReLULayer::backward(){
  return;
}

ReLULayer::~ReLULayer(){
  delete CudaM;
}