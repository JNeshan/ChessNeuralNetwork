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
  if(size == -1){ //blank tensor
    return;
  }
  if(device == TensorLocation::GPU){ //lazy delete
    TryCuda(cudaFree(data));
    size = -1;
    return;
  }
  delete[] data;
}

Tensor::Tensor() : size(-1), device(TensorLocation::CPU), data(nullptr), dimensions() {}

Tensor::Tensor(const std::vector<int>& d, const TensorLocation loc = TensorLocation::CPU){
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

  while(dimensions.size() < 4){
    dimensions.push_back(1);
  }

  if(loc == TensorLocation::CPU){
    device = TensorLocation::CPU;
    data = new float[size];
  }
  else{
    device = TensorLocation::GPU;
    TryCuda(cudaMalloc((void**)&data, size * sizeof(float)));
  }
}

Tensor::Tensor(Tensor&& r) noexcept : dimensions(r.dimensions), size(r.size), data(r.data), device(r.device){
  r.data = nullptr;
  r.size = 0;
}

Tensor& Tensor::operator=(Tensor&& r) noexcept{
  if(this == &r){
    return *this;
  }
  if(data){
    if(device == TensorLocation::GPU){
      TryCuda(cudaFree(&data));
    }
    else{
      delete[] data;
    }
  }
  dimensions = r.dimensions;
  size = r.size;
  device = r.device;
  data = r.data;
  r.data = nullptr;
  r.size = 0;
  return *this;
}

Tensor& Tensor::operator=(const Tensor& r){
  if(this == &r){
    return *this;
  }
  
  if(data != nullptr){
    if(device == TensorLocation::GPU){
      TryCuda(cudaFree(data));
    }
    else{
      delete[] data;
    }
  }

  size = r.size;
  device = r.device;
  dimensions = r.dimensions;
  data = nullptr;
  if(device == TensorLocation::GPU){
    TryCuda(cudaMalloc((void**)&data, size * sizeof(float)));
    TryCuda(cudaMemcpy(data, r.data, size * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  else{
    data = new float[size];
    memcpy(data, r.data, size * sizeof(float));
  }
}

float* Tensor::cpuData() const{
  if(device == TensorLocation::GPU){
    throw std::invalid_argument("Data stored on GPU");
  }
  else if(data == nullptr){
    throw std::invalid_argument("Data not initialized");
  }
  return data;
}

float* Tensor::gpuData() const{
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

void Tensor::flatten(){
  if(dimensions[0] == 1 && dimensions[1] == 1){
    return; //already 2d
  }
  int s = size / dimensions[0];
  dimensions[2] = dimensions[0];
  dimensions[3] = size / dimensions[0];
  dimensions[0] = 1; dimensions[1] = 1;
  return;
}

void Tensor::reshape(const std::vector<int>& dim){
  int s = 1;
  for(auto d : dim) s *= d;
  if(s != size){
    throw("Invalid reshape size");
  }
  dimensions = dim;
}