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
  cudnnTensorDescriptor_t tensorD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  }

  ~CudaMembers(){
    TryCuda(cudnnDestroyTensorDescriptor(tensorD));
    TryCuda(cudnnDestroy(handle));
  };
};

SoftmaxLayer::SoftmaxLayer() : outFeat(4672){
  CudaM = new CudaMembers();
}

SoftmaxLayer::~SoftmaxLayer(){
  delete CudaM;
}

std::pair<Tensor, std::unique_ptr<ForwardCache>> SoftmaxLayer::forward(const Tensor& T){
  if(T.n != 2){ //input must be 2 dimensional
    throw("Softmax input invalid n = " + std::to_string(T.n));
  }
  Tensor({T.dimensions[0], outFeat}, TensorLocation::GPU);
  output = Tensor({T.dimensions[0], outFeat}, TensorLocation::GPU); //storing output for back
  if(output.size != T.size){ //check to ensure the matrices are the same size (also means 2nd dimensions are equal)
    throw("Bad softmax input"); 
  }
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], 1, 1)); //input descriptor
  TryCuda(cudnnSoftmaxForward(CudaM->handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &mx, CudaM->tensorD, T.gpuData(), &mn, CudaM->tensorD, output.gpuData()));
  return output;
}

std::pair<Tensor, std::unique_ptr<BackwardCache>> SoftmaxLayer::backward(const Tensor& gradient, const ForwardCache& fCache){
  if(gradient.n != 2 || gradient.size < output.size){
    throw("Softmax recieved bad gradient or recorded faulty output");
  }
  Tensor iGrad(output.dimensions, TensorLocation::GPU, output.n);
  TryCuda(cudnnSoftmaxBackward(CudaM->handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &mx, CudaM->tensorD, 
                              output.gpuData(), CudaM->tensorD, gradient.gpuData(), &mn, CudaM->tensorD, iGrad.gpuData()));
  CudaM->resetTemp();
  return iGrad;
}
