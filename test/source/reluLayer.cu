#include "../header/reluLayer.h"
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

ReLULayer::ReLULayer(){
  TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  TryCuda(cudnnCreateActivationDescriptor(&reLU));  
  TryCuda(cudnnSetActivationDescriptor(reLU, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));


}

Tensor ReLULayer::forward(const Tensor& T, bool train){
  
  Tensor output(T.dimensions, TensorLocation::GPU, T.n);
  TryCuda(cudnnSetTensor4dDescriptor(tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.size, 1, 1, 1));
  TryCuda(cudnnActivationForward(nnHandle, reLU, &mx, tensorD, T.gpuData(), &mn, tensorD, output.gpuData()));
  return output;
}

Tensor ReLULayer::backward(const Tensor& gradient){
  Tensor iGrad(input.dimensions, TensorLocation::GPU, input.n);
  TryCuda(cudnnActivationBackward(nnHandle, reLU, &mx, tensorD, gradient.gpuData(), tensorD, gradient.gpuData(), tensorD, input.gpuData(), &mn, tensorD, iGrad.gpuData()));  
  return iGrad;
}

void ReLULayer::saveTensor(std::ofstream& oF){
  return;
}
void ReLULayer::genTensorData(){
  return;
}
void ReLULayer::loadTensor(std::ifstream& iF){
  return;
}

void ReLULayer::cleanSave(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }
  input.writeTensor(oF);
}


ReLULayer::~ReLULayer(){
  TryCuda(cudnnDestroyTensorDescriptor(tensorD));
  TryCuda(cudnnDestroyActivationDescriptor(reLU));
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> ReLULayer::getLearningData(){
  return {};
}