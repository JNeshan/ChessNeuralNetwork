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
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t actD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&tensorD));
    TryCuda(cudnnCreateActivationDescriptor(&actD));

  }

  void resetTemp(){
    TryCuda(cudnnDestroyTensorDescriptor(tensorD));
    TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  }

  ~CudaMembers(){
    TryCuda(cudnnDestroyTensorDescriptor(tensorD));
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

std::pair<Tensor, std::unique_ptr<ForwardCache>> tanhLayer::forward(const Tensor& T){
  Tensor output(T.dimensions, TensorLocation::GPU, T.n);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.size, 1, 1, 1));
  TryCuda(cudnnActivationForward(CudaM->handle, CudaM->actD, &mx, CudaM->tensorD, T.gpuData(), &mn, CudaM->tensorD, output.gpuData()));
  ForwardCache(output);
  return output;
}

std::pair<Tensor, std::unique_ptr<BackwardCache>> tanhLayer::backward(const Tensor& gradient, const ForwardCache& fCache){
  Tensor iGrad(dimensions, TensorLocation::GPU, n);
  TryCuda(cudnnActivationBackward(CudaM->handle, CudaM->actD, &mx, CudaM->outputD, output.gpuData(), CudaM->outputD, 
                                  gradient.gpuData(), CudaM->inputD, output.gpuData(), &mn, CudaM->inputD, iGrad.gpuData()));

  CudaM->resetTemp();
  return iGrad; 
}