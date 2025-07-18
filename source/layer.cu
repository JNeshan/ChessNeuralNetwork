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

thread_local cudnnHandle_t Layer::nnHandle = nnCreate();
thread_local cublasHandle_t Layer::blasHandle = blasCreate();

Layer::Layer(){
}

Layer::~Layer(){
  
}