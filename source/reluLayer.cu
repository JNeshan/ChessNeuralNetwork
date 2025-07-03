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
  cudnnTensorDescriptor_t inputD, outputD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
    TryCuda(cudnnCreateActivationDescriptor(&reLU));

  }
  ~CudaMembers(){
    TryCuda(cudnnDestroyTensorDescriptor(inputD));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnDestroyActivationDescriptor(reLU));
    TryCuda(cudnnDestroy(handle));
  };
  void resetTemp(){
    TryCuda(cudnnDestroyTensorDescriptor(inputD));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
  }
};

ReLULayer::ReLULayer(){
  CudaM = new CudaMembers();
  TryCuda(cudnnSetActivationDescriptor(CudaM->reLU, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 2));

}

Tensor ReLULayer::forward(const Tensor& T){
  Tensor output(T.dimensions, TensorLocation::GPU);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], T.dimensions[2], T.dimensions[3]));
  TryCuda(cudnnActivationForward(CudaM->handle, CudaM->reLU, &alpha, CudaM->inputD, T.gpuData(), &beta, CudaM->outputD, output.gpuData()));
  CudaM->resetTemp();
  return output;
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> ReLULayer::backward(const Tensor& gradient){
  iGrad = Tensor(input.dimensions, TensorLocation::GPU);
  TryCuda(cudnnActivationBackward(CudaM->handle, CudaM->reLU, &mx, CudaM->outputD, gradient.gpuData(), CudaM->outputD, gradient.gpuData(), CudaM->inputD, input.gpuData(), &mn, CudaM->inputD, iGrad.gpuData()));
  
  CudaM->resetTemp();
  return {{&input}, {&iGrad}};
}

ReLULayer::~ReLULayer(){
  delete CudaM;
}