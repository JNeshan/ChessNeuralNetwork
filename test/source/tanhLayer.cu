#include "../header/tanhLayer.h"
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

tanhLayer::tanhLayer(){
  TryCuda(cudnnCreateTensorDescriptor(&tensorD));
  TryCuda(cudnnCreateActivationDescriptor(&actD));
  TryCuda(cudnnSetActivationDescriptor(actD, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0f));
}

tanhLayer::tanhLayer(const tanhLayer& lay){
  TryCuda(cudnnCreateTensorDescriptor(&this->tensorD));
  TryCuda(cudnnCreateActivationDescriptor(&this->actD));
  TryCuda(cudnnSetActivationDescriptor(this->actD, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0f));
}

std::unique_ptr<Layer> tanhLayer::clone(){
  return(std::make_unique<tanhLayer>(*this));
}

tanhLayer::~tanhLayer(){
  TryCuda(cudnnDestroyTensorDescriptor(tensorD));
  TryCuda(cudnnDestroyActivationDescriptor(actD));
}

Tensor tanhLayer::forward(Tensor<__half>& T, bool train){
  
  TryCuda(cudnnSetTensor4dDescriptor(tensorD, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, T.size, 1, 1, 1));
  TryCuda(cudnnActivationForward(nnHandle, actD, &mx, tensorD, T.gpuData(), &mn, tensorD, T.gpuData()));
  if(train){
    this->output = T;
  }
  return std::move(T);
}

Tensor tanhLayer::backward(Tensor<__half>& gradient){
  Tensor iGrad(output.dimensions, TensorLocation::GPU, output.n);
  TryCuda(cudnnActivationBackward(nnHandle, actD, &mx, tensorD, output.gpuData(), tensorD, 
                                  gradient.gpuData(), tensorD, output.gpuData(), &mn, tensorD, iGrad.gpuData()));

  return std::move(iGrad); 
}

void tanhLayer::saveTensor(std::ofstream& oF){
  return;
}
void tanhLayer::genTensorData(){
  return;
}
void tanhLayer::loadTensor(std::ifstream& iF){
  return;
}

void tanhLayer::cleanSave(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }
  oF <<"tanh Layer Tensor:\n";
  output.writeTensor(oF);
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> tanhLayer::getLearningData(){
  return {};
}