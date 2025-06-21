#include <stdexcept>
#include <vector>
#include "header/tensor.h"
#include "cuda_runtime.h"

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

Tensor::~Tensor() {
  if(device == TensorLocation::GPU){ //lazy delete
    TryCuda(cudaFree(data));
    return;
  }
  delete[] data;
}

Tensor::Tensor(const std::vector<int>& d){
  if(d.size() == 0){
    throw std::invalid_argument("Tensor dimensions empty");
  }
  size = 1;
  dimensions = std::vector<int>();
  for(auto dX : d){
    if(dX <= 0){
      throw std::invalid_argument("Bad dimension size");
    }
    dimensions.push_back(dX);
    size *= dX;
  }
  device = TensorLocation::CPU;
  data = new float[size];
}

float* Tensor::cpuData(){
  if(device == TensorLocation::GPU){
    throw std::invalid_argument("Data stored on GPU");
  }
  else if(data == nullptr){
    throw std::invalid_argument("Data not initialized");
  }
  return data;
}

float* Tensor::gpuData(){
  if(device == TensorLocation::CPU){
    throw std::invalid_argument("Data stored on CPU");
  }
  else if(data == nullptr){
    throw std::invalid_argument("Data not initialized");
  }
  return data;
}

void Tensor::cpuSend(){
  if(device == TensorLocation::CPU){
    return;
  }
  if(data == nullptr){
    throw(std::invalid_argument("Tensor is unintialized"));
  }
  float* tmpData = new float[size];
  TryCuda(cudaMemcpy(tmpData, data, size * sizeof(float), cudaMemcpyDeviceToHost));
  TryCuda(cudaFree(data));
  data = tmpData;
  device = TensorLocation::CPU;
}

void Tensor::gpuSend(){
  if(device == TensorLocation::GPU){
    return; //already in gpu memory
  }
  if(data == nullptr){
    throw(std::invalid_argument("Data is not populated and device is on GPU"));
  }
  float* tmpData;
  TryCuda(cudaMalloc((void**)&tmpData, size * sizeof(float))); //allocating memory within the GPU
  TryCuda(cudaMemcpy(tmpData, data, size * sizeof(float), cudaMemcpyHostToDevice));  //copying into GPU memory
  
  delete[] data;
  data = tmpData;
  device = TensorLocation::GPU;
}