#include "../header/layer.h"

cudnnHandle_t nnCreate(){
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  return handle;
}

cublasHandle_t blasCreate(){
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  return handle;
}

ForwardPackage::ForwardPackage(Tensor<__half>& o, std::vector<Tensor<__half>>& error) : output(std::move(o)), errSet(std::move(error)) {}
ForwardPackage::ForwardPackage(Tensor<__half>& o) : output(std::move(o)), errSet({}) {}

BackwardPackage::BackwardPackage(Tensor<__half>& o, std::vector<Tensor<__half>>& gs) : iGrad(std::move(o)), trainGrads(std::move(gs)) {}
BackwardPackage::BackwardPackage(Tensor<__half>& o) : iGrad(std::move(o)), trainGrads({}) {}


thread_local cudnnHandle_t Layer::nnHandle = nnCreate();
thread_local cublasHandle_t Layer::blasHandle = blasCreate();
void* Layer::wsPtr = nullptr;
size_t Layer::wsSize = 0;

Layer::Layer(){
}

Layer::~Layer(){
  
}

