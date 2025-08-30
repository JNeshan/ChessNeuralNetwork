#include "../header/tanhLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>

thread_local cudnnHandle_t Layer::nnHandle{};
thread_local cublasHandle_t Layer::blasHandle{};


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

tanhLayer::tanhLayer(){
  TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  TryCuda(cudnnCreateActivationDescriptor(&actD));
  TryCuda(cudnnSetActivationDescriptor(actD, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0f));
}

tanhLayer::~tanhLayer(){
  TryCuda(cudnnDestroyTensorDescriptor(tensorD));
  TryCuda(cudnnDestroyActivationDescriptor(actD));
}

Tensor tanhLayer::forward(const Tensor& T, bool train){
  Tensor output(T.dimensions, TensorLocation::GPU, T.n);
  TryCuda(cudnnSetTensor4dDescriptor(tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.size, 1, 1, 1));
  TryCuda(cudnnActivationForward(nnHandle, actD, &mx, tensorD, T.gpuData(), &mn, tensorD, output.gpuData()));
  return output;
}

Tensor tanhLayer::backward(const Tensor& gradient){
  Tensor iGrad(output.dimensions, TensorLocation::GPU, output.n);
  TryCuda(cudnnActivationBackward(nnHandle, actD, &mx, tensorD, output.gpuData(), tensorD, 
                                  gradient.gpuData(), tensorD, output.gpuData(), &mn, tensorD, iGrad.gpuData()));

  return iGrad; 
}