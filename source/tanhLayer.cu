#include "header/tanhLayer.h"
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
  cudnnActivationDescriptor_t actD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
    TryCuda(cudnnCreateActivationDescriptor(&actD));

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
    TryCuda(cudnnDestroyActivationDescriptor(actD));
    TryCuda(cudnnDestroy(handle));
  };
};

tanhLayer::tanhLayer(){
  CudaM = new CudaMembers();
  TryCuda(cudnnSetActivationDescriptor(CudaM->actD, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0f));

}

tanhLayer::~tanhLayer(){
  delete CudaM;
}

Tensor tanhLayer::forward(const Tensor& T){
  output = Tensor({T.dimensions[0], 1}, TensorLocation::GPU);
  input = Tensor(T);

  if(T.size != output.size){
    throw("Bad tanh input");
  }

  TryCuda(cudnnSetTensor4dDescriptor(CudaM->inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], 1, 1, 1));
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], 1, 1, 1));
  TryCuda(cudnnActivationForward(CudaM->handle, CudaM->actD, &mx, CudaM->inputD, input.gpuData(), &mn, CudaM->outputD, output.gpuData()));
  return output;
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> tanhLayer::backward(const Tensor& gradient){
  iGrad = Tensor(input.dimensions);
  TryCuda(cudnnActivationBackward(CudaM->handle, CudaM->actD, &mx, CudaM->outputD, output.gpuData(), CudaM->outputD, 
                                  gradient.gpuData(), CudaM->inputD, output.gpuData(), &mn, CudaM->inputD, iGrad.gpuData()));

  CudaM->resetTemp();
  return {{&input}, {&iGrad}}; 
}