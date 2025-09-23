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

ReLULayer::ReLULayer(const ReLULayer& lay){
  TryCuda(cudnnCreateTensorDescriptor(&this->tensorD));
  TryCuda(cudnnCreateActivationDescriptor(&this->reLU));
  TryCuda(cudnnSetActivationDescriptor(this->reLU, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
}

std::unique_ptr<Layer> ReLULayer::clone(){
  return(std::make_unique<ReLULayer>(*this));
}


Tensor ReLULayer::forward(Tensor& T, bool train){
  if(train){
    this->input = T;
  }
  auto start = std::chrono::steady_clock::now();

  auto elapsed = std::chrono::steady_clock::now() - start;
  TryCuda(cudnnSetTensor4dDescriptor(tensorD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.size, 1, 1, 1));
  TryCuda(cudnnActivationForward(nnHandle, reLU, &mx, tensorD, T.gpuData(), &mn, tensorD, T.gpuData()));
  
  elapsed = std::chrono::steady_clock::now() - start;
  ////std::cout<<std::string("Time in relu: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  return T;
}

Tensor ReLULayer::backward(Tensor& gradient){
  //Tensor iGrad(input.dimensions, TensorLocation::GPU, input.n);
  TryCuda(cudnnActivationBackward(nnHandle, reLU, &mx, tensorD, gradient.gpuData(), tensorD, gradient.gpuData(), tensorD, input.gpuData(), &mn, tensorD, gradient.gpuData()));  
  return gradient;
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