#include "header/reluLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

void ReLULayer::forward(Tensor T){
  
  
}

void ReLULayer::backward(){
  return;
}
