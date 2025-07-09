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
  cudnnTensorDescriptor_t tensorD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&tensorD));
    TryCuda(cudnnCreateActivationDescriptor(&reLU));

  }
  ~CudaMembers(){
    TryCuda(cudnnDestroyTensorDescriptor(tensorD));
    TryCuda(cudnnDestroyActivationDescriptor(reLU));
    TryCuda(cudnnDestroy(handle));
  }
  void resetTemp(){
    TryCuda(cudnnDestroyTensorDescriptor(tensorD));
    TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  }
};

ReLULayer::ReLULayer(){
  CudaM = new CudaMembers();
  TryCuda(cudnnSetActivationDescriptor(CudaM->reLU, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 2));

}

std::pair<Tensor, std::unique_ptr<ForwardCache>> ReLULayer::forward(const Tensor& T){
  Tensor output(T.dimensions, TensorLocation::GPU, T.size);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.size, 1, 1, 1));
  TryCuda(cudnnActivationForward(CudaM->handle, CudaM->reLU, &alpha, CudaM->tensorD, T.gpuData(), &beta, CudaM->tensorD, output.gpuData()));
  return output;
}

std::pair<Tensor, std::unique_ptr<BackwardCache>> ReLULayer::backward(const Tensor& gradient, const ForwardCache& fCache){
  Tensor iGrad(input.dimensions, TensorLocation::GPU, input.n);
  TryCuda(cudnnActivationBackward(CudaM->handle, CudaM->reLU, &mx, CudaM->tensorD, gradient.gpuData(), CudaM->tensorD, gradient.gpuData(), CudaM->tensorD, input.gpuData(), &mn, CudaM->tensorD, iGrad.gpuData()));  
  BackwardCache back();
  std::pair<std::vector<Tensor*>, std::vector<Tensor*>>
  return iGrad;
}

ReLULayer::~ReLULayer(){
  delete CudaM;
}