#include <cuda_runtime.h>
#include "../header/optimizer.h"

//should be functional, still needs to verify it didn't create any cuda errors

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__global__ void GradDescentKernel(float* in, const float* grad, const float lR, const int size){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size){
    in[idx] -= lR * grad[idx];
  }
}

Optimizer::Optimizer(const float rate) : lR(rate){}

Optimizer::~Optimizer(){}

void Optimizer::setRate(const float rate){
  this->lR = rate;
}

void Optimizer::optimize(const Tensor& in, const Tensor& grad){
  //batches is always stored in dimensions[0]
  const int thCount = 256, m = ((in.size + thCount - 1) / thCount); //number of threads per thread block, number of blocks in 1d grid
  dim3 blockDim(thCount);
  dim3 gridDim(m);
  GradDescentKernel<<<gridDim, blockDim>>>(in.gpuData(), grad.gpuData(), lR, in.size);
  return;
}

void Optimizer::batchOptimize(std::pair<std::vector<Tensor*>, std::vector<Tensor*>>& trainingBatch){
  auto [Main, Grad] = trainingBatch;
  for(int i = 0; i < Main.size(); i++){
    Tensor* in = trainingBatch.first[i], *grad = trainingBatch.second[i];
    const int thCount = 256, m = ((in->size + thCount - 1) / thCount); //number of threads per thread block, number of blocks in 1d grid
    dim3 blockDim(thCount);
    dim3 gridDim(m);
    GradDescentKernel<<<gridDim, blockDim>>>(in->gpuData(), grad->gpuData(), lR, in->size);
  }
}