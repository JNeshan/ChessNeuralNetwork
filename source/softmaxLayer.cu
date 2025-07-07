#include "header/softmaxLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>

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
  cudnnTensorDescriptor_t inputD, outputD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
  }

  void resetTemp(){
    TryCuda(cudnnDestroyTensorDescriptor(inputD));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
  }

  ~CudaMembers(){
    TryCuda(cudnnDestroyTensorDescriptor(inputD));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnDestroy(handle));
  };
};

SoftmaxLayer::SoftmaxLayer() : outFeat(4672){
  CudaM = new CudaMembers();
}

SoftmaxLayer::~SoftmaxLayer(){
  delete CudaM;
}

Tensor SoftmaxLayer::forward(const Tensor& T){
  output = Tensor({1, 1, input.dimensions[0], outFeat}, TensorLocation::GPU);
  if(output.size != T.size){
    throw("Bad softmax input");
  }
  input = Tensor(T);
  iGrad = Tensor(input.dimensions, TensorLocation::GPU);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.dimensions[0], outFeat, 1, 1));
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.dimensions[0], input.size / input.dimensions[0], 1, 1));
  TryCuda(cudnnSoftmaxForward(CudaM->handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &mx, CudaM->inputD, input.gpuData(), &mn, CudaM->outputD, output.gpuData()));
  return output;
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> SoftmaxLayer::backward(const Tensor& gradient){
  
  Tensor grad(input.dimensions, TensorLocation::GPU);
  TryCuda(cudnnSoftmaxBackward(CudaM->handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &mx, CudaM->outputD, 
                              output.gpuData(), CudaM->outputD, gradient.gpuData(), &mn, CudaM->inputD, iGrad.gpuData()));
  
  CudaM->resetTemp();
  return {{&input}, {&iGrad}};
}
