#include "../header/softmaxLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>
#include <string>

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

struct CudaCache{
  cudnnTensorDescriptor_t tensorD;
  CudaCache(){
    TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  }
  ~CudaCache(){
    TryCuda(cudnnDestroyTensorDescriptor(tensorD));
  }
};

SoftmaxLayer::SoftmaxLayer() : outFeat(4672){
  TryCuda(cudnnCreateTensorDescriptor(&tensorD));
}

SoftmaxLayer::~SoftmaxLayer(){
  TryCuda(cudnnDestroyTensorDescriptor(tensorD));
}

Tensor SoftmaxLayer::forward(const Tensor& T){

  if(T.n != 2){ //input must be 2 dimensional
    throw("Softmax input invalid n = " + std::to_string(T.n));
  }
  Tensor({T.dimensions[0], outFeat}, TensorLocation::GPU);
  output = Tensor({T.dimensions[0], outFeat}, TensorLocation::GPU); //storing output for back
  if(output.size != T.size){ //check to ensure the matrices are the same size (also means 2nd dimensions are equal)
    throw("Bad softmax input"); 
  }
  TryCuda(cudnnSetTensor4dDescriptor(tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], 1, 1)); //input descriptor
  TryCuda(cudnnSoftmaxForward(nnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &mx, tensorD, T.gpuData(), &mn, tensorD, output.gpuData()));
  return output;
}

Tensor SoftmaxLayer::backward(const Tensor& gradient){
  if(gradient.n != 2 || gradient.size < output.size){
    throw("Softmax recieved bad gradient or recorded faulty output");
  }
  Tensor iGrad(output.dimensions, TensorLocation::GPU, output.n);
  TryCuda(cudnnSoftmaxBackward(nnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &mx, tensorD, 
                              output.gpuData(), tensorD, gradient.gpuData(), &mn, tensorD, iGrad.gpuData()));
  return iGrad;
}

void SoftmaxLayer::saveTensor(std::ofstream& oF){
  return;
}
void SoftmaxLayer::genTensorData(){
  return;
}
void SoftmaxLayer::loadTensor(std::ifstream& iF){
  return;
}

void SoftmaxLayer::cleanSave(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }
  oF << "Softmax Layer Tensor:\n";
  output.writeTensor(oF);
}