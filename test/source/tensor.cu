#include <stdexcept>
#include <vector>
#include "../header/tensor.h"
#include "cuda_runtime.h"
#include <string>
#include <sstream>
#include <iomanip>

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ std::string formatFloat(float f, int width, int precision){
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision)<<f;
  std::string out = oss.str();

  if(out.length() < width){
    out = std::string(width - out.length(), ' ') + out;
  }
  else if(out.length() > width){
    out = out.substr(0, width);
  }
  return out;
}

__global__ void AddKernel(const float* A, const float* B, float* out, const int n, const int m){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n * m){
    if(A == out){
      out[idx] += B[idx];
    }
    else{
      out[idx] = B[idx] + A[idx];
    }
  }
}

Tensor::~Tensor() {
  if(this->size <= 0 || this->data == nullptr){ //blank tensor
    return;
  }
  if(this->device == TensorLocation::GPU){ //lazy delete
    TryCuda(cudaFree(this->data));
    this->size = -1;
    return;
  }

  delete[] this->data;
}

Tensor::Tensor() : size(-1), device(TensorLocation::CPU), data(nullptr), dimensions() {}

Tensor::Tensor(const std::vector<int>& dim, const TensorLocation loc, int nth) : n(nth), dimensions(dim), device(loc){
  if(dim.size() == 0){
    throw std::invalid_argument("Tensor dimensions empty");
  }
  if(!nth){ //nth defaults to 0 if no dimension count is given so dim can instead relay the proper n as long as it is not padded
    this->n = dim.size();
  }

  this->size = 1;
  for(auto d : dim){
    if(d < 1) throw std::runtime_error("Non-positive dimension size");
    this->size *= d;
  }

  while(this->dimensions.size() < 4 || this->dimensions.size() < n){
    this->dimensions.push_back(1);
  }

  if(loc == TensorLocation::CPU){
    this->device = TensorLocation::CPU;
    this->data = new float[this->size]();
  }
  else{
    device = TensorLocation::GPU;
    TryCuda(cudaMalloc((void**)&data, this->size * sizeof(float)));
  }
}

Tensor::Tensor(const Tensor& r){
  this->device = r.device;
  this->dimensions = r.dimensions;
  this->n = r.n;
  this->size = r.size;
  if(r.data != nullptr){
    if(this->device == TensorLocation::GPU){
      TryCuda(cudaMalloc((void**)&data, size * sizeof(float)));
      TryCuda(cudaMemcpy(this->data, r.data, size * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    else{
      data = new float[size];
      memcpy(data, r.data, size * sizeof(float));
    }
  }
}

Tensor::Tensor(Tensor&& r) : dimensions(r.dimensions), size(r.size), data(r.data), device(r.device), n(r.n){
  r.data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& r){
  if(this == &r){
    return *this;
  }

  if(this->data){
    if(this->device == TensorLocation::GPU){
      TryCuda(cudaFree(this->data));
    }
    else{
      delete[] this->data;
    }
  }

  this->size = r.size;
  this->device = r.device;
  this->dimensions = r.dimensions;
  this->n = r.n;
  this->data = r.data;
  r.data = nullptr; 
  return *this;
}

Tensor& Tensor::operator=(const Tensor& r){
  if(this == &r){
    return *this;
  }
  
  if(this->data != nullptr){
    if(this->device == TensorLocation::GPU){
      TryCuda(cudaFree(data));
    }
    else{
      delete[] this->data;
    }
  }

  this->size = r.size;
  this->device = r.device;
  this->dimensions = r.dimensions;
  this->n = r.n;
  this->data = nullptr;
  if(this->device == TensorLocation::GPU){
    TryCuda(cudaMalloc((void**)&data, this->size * sizeof(float)));
    TryCuda(cudaMemcpy(data, r.data, this->size * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  else{
    this->data = new float[this->size];
    memcpy(this->data, r.data, this->size * sizeof(float));
  }
  return *this;
}

float* Tensor::cpuData() const{
  if(this->device == TensorLocation::GPU){
    throw std::invalid_argument("Data stored on GPU");
  }
  else if(data == nullptr){
    throw std::invalid_argument("Data not initialized");
  }
  return data;
}

float* Tensor::gpuData() const{
  if(this->device == TensorLocation::CPU){
    throw std::invalid_argument("Data stored on CPU");
  }
  else if(data == nullptr){
    throw std::invalid_argument("Data not initialized");
  }
  return data;
}

void Tensor::cpuSend(){
  if(this->device == TensorLocation::CPU){
    //std::cout<<"Already on CPU"<<std::endl;
    return;
  }
  if(this->data == nullptr){
    throw std::invalid_argument("Data has not been allocated");
  }
  if(this->size <= 0){
    this->device == TensorLocation::CPU;
    return;
  }
  float* tmpData = new float[this->size];
  TryCuda(cudaMemcpy(tmpData, this->data, this->size * sizeof(float), cudaMemcpyDeviceToHost));
  TryCuda(cudaFree(data));
  this->data = tmpData;
  tmpData = nullptr;
  this->device = TensorLocation::CPU;
}

void Tensor::gpuSend(){

  if(this->device == TensorLocation::GPU){
   // std::cout<<"Tensor already in GPU memory"<<std::endl;
    return; //already in gpu memory
  }
  if(this->data == nullptr){
    throw std::invalid_argument("Data has not been allocated");
  }
  if(this->size <= 0){
    this->device == TensorLocation::GPU;
    return;
  }
  float* tmpData;
  TryCuda(cudaMalloc((void**)&tmpData, size * sizeof(float))); //allocating memory within the GPU
  TryCuda(cudaMemcpy(tmpData, data, size * sizeof(float), cudaMemcpyHostToDevice));  //copying into GPU memory
  
  delete[] this->data;
  this->data = tmpData;
  tmpData = nullptr;
  this->device = TensorLocation::GPU;
}

void Tensor::reshape(const std::vector<int>& dim, const int nth){
  int s = 1;
  for(auto d : dim){
    if(d < 1){
      throw std::runtime_error("Non-positive dimension rehsape size");
    }
    s *= d;
  }

  if(s != size){ //the new shape must have the same size as the previous to be valid
    throw std::runtime_error("Different reshape size");
  }
  this->n = nth;
  this->dimensions = dim;
  while(this->dimensions.size() < 4){
    this->dimensions.push_back(1);
  }
}

Tensor Tensor::segment(const int n_i){
  try
  {
    if(n_i < 0 || n_i > this->dimensions[0]){
      throw std::runtime_error("Bad batch index to copy");
    }
      std::vector<int> tDim = this->dimensions;
      tDim[0] = 1;
      Tensor T(tDim, this->device, this->n);
      int ind = n_i * (this->size / this->dimensions[0]);
      if(this->device == TensorLocation::CPU){
        
        memcpy(T.cpuData(), this->data + ind, T.size * sizeof(float));
      }
      else{
        TryCuda(cudaMemcpy(T.gpuData(), this->data + ind, T.size * sizeof(float), cudaMemcpyDeviceToDevice));
      }
    return std::move(T);
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  return Tensor();
}
//dont use
Tensor::Tensor(std::vector<Tensor>& t) : dimensions(t[0].dimensions), size(t[0].size), n(t[0].n){
  this->dimensions[0] = t.size();
  this->size *= this->dimensions[0];
}

void Tensor::batchBuild(const std::vector<Tensor*>& t, const TensorLocation loc){
  if(t.size() == 0){
    throw std::runtime_error("Empty batch vector");
  }
  this->dimensions = t[0]->dimensions;
  this->dimensions[0] = t.size();
  this->size = t[0]->size * t.size();
  this->n = t[0]->n;
  this->device = loc;
  if(loc == TensorLocation::CPU){
    this->data = new float[this->size];
    int i = 0;
    for(auto tensor : t){
      auto idx = this->data + (i * tensor->size);
      if(tensor->device == TensorLocation::CPU){
        memcpy(idx, tensor->cpuData(), tensor->size * sizeof(float));
      }
      else{
        TryCuda(cudaMemcpy(idx, tensor->gpuData(), tensor->size * sizeof(float), cudaMemcpyDeviceToHost));
      }
      i++;
    }
  }
  else{
    TryCuda(cudaMalloc((void**)&this->data, this->size * sizeof(float)));
    int i = 0;
    for(auto tensor : t){
      auto idx = this->data + (i * tensor->size);
      auto cudaCpy = tensor->device == TensorLocation::GPU ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
      TryCuda(cudaMemcpy(idx, tensor->data, tensor->size * sizeof(float), cudaCpy));
      i++;
    }
  }
}

void Tensor::batchBuild(const std::vector<Tensor>& t, const TensorLocation loc){
  if(t.size() == 0){
    throw std::runtime_error("Empty batch vector");
  }
  this->dimensions = t[0].dimensions;
  this->dimensions[0] = t.size();
  this->size = t[0].size * t.size();
  this->n = t[0].n;
  this->device = loc;
  if(loc == TensorLocation::CPU){
    this->data = new float[this->size];
    int i = 0;
    for(auto& tensor : t){
      auto idx = this->data + (i * tensor.size);
      if(tensor.device == TensorLocation::CPU){
        memcpy(idx, tensor.cpuData(), tensor.size * sizeof(float));
      }
      else{
        TryCuda(cudaMemcpy(idx, tensor.gpuData(), tensor.size * sizeof(float), cudaMemcpyDeviceToHost));
      }
      i++;
    }
  }
  else{
    TryCuda(cudaMalloc((void**)&this->data, this->size * sizeof(float)));
    int i = 0;
    for(auto tensor : t){
      auto idx = this->data + (i * tensor.size);
      auto cudaCpy = tensor.device == TensorLocation::GPU ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
      TryCuda(cudaMemcpy(idx, tensor.data, tensor.size * sizeof(float), cudaCpy));
      i++;
    }
  }
}
//appending
void Tensor::writeTensor(std::ofstream& oF){
  int width = 16, precision = 14;
  if(!oF.is_open()){
    std::cout<<"Output stream not open"<<std::endl;
    return;
  }
  if(data == nullptr || size <= 0){
    std::cout<<"Tensor not allocated"<<std::endl;
    return;
  }
  if(device == TensorLocation::GPU){
    this->cpuSend();
  }
  oF << std::to_string(this->n) << "-dimensional tensor " << this->dimensions[0];
  for(int i = 1; i < this->n; i++){
    oF << " x " << this->dimensions[i];
  }
  oF << "\n";
  int x = this->dimensions[0], y = this->size / x;
  for(int i = 0; i < x; i++){
    for(int j = 0; j < y; j++){
      std::string value = formatFloat(data[i*y + j], width, precision);
      oF << value << " ";
    }
    oF << "\n";
  }
  oF << "\n";
}

void Tensor::writeBinary(std::ofstream& oF){
  if(data == nullptr){
    std::cout<<"Tensor not populated"<<std::endl;
    return;
  }
  if(!oF.is_open()){
    std::cout<<"Output not open"<<std::endl;
    return;
  }
  if(this->device == TensorLocation::GPU){
    this->cpuSend();
  }
  oF.write(reinterpret_cast<const char*>(&this->size), sizeof(int));
  oF.write(reinterpret_cast<const char*>(&this->n), sizeof(int));
  oF.write(reinterpret_cast<const char*>(this->dimensions.data()), this->dimensions.size() * sizeof(int));
  oF.write(reinterpret_cast<const char*>(this->data), this->size * sizeof(float));
}

void Tensor::readBinary(std::ifstream& iF){
  if(!iF.is_open()){
    std::cout<<"File not open to read"<<std::endl;
    return;
  }
  if(this->data == nullptr){
    std::cout<<std::endl;
  }
  else if(this->device == TensorLocation::GPU){
    TryCuda(cudaFree(data));
  }
  else{
    delete[] data;
  }
  iF.read(reinterpret_cast<char*>(&this->size), sizeof(int));
  iF.read(reinterpret_cast<char*>(&this->n), sizeof(int));
  int d = std::max(4, this->n);
  dimensions = std::vector<int>(d);
  data = new float[this->size];
  iF.read(reinterpret_cast<char*>(this->dimensions.data()), d * sizeof(int));
  iF.read(reinterpret_cast<char*>(this->data), this->size * sizeof(float));
  this->device = TensorLocation::CPU;
}

void Tensor::gpuAdd(Tensor& B){

  if(this->size != B.size){
    throw std::runtime_error("Different sizes for addition");
  }
  this->gpuSend();
  B.gpuSend();
  
  float* bData = B.gpuData();
  float* aData = this->gpuData();

  int n = this->dimensions[0], m = this->size / n, thrdCnt = 256;
  dim3 gridDim((this->size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);

  AddKernel<<<gridDim, blockDim>>>(aData, bData, aData, n, m);
}

float* Tensor::gpuDataForce(){
  this->gpuSend();
  return this->data;
}

float* Tensor::cpuDataForce(){
  this->cpuSend();
  return this->data;
}