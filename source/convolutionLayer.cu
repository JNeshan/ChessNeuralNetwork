#include "header/convolutionLayer.h"
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

struct CudaMembers{
  cudnnHandle_t* handle;
  
  cudnnTensorDescriptor_t inputD, outputD, biasD;
  cudnnFilterDescriptor_t filterD;
  cudnnConvolutionDescriptor_t convoD;

  CudaMembers(){
    TryCuda(cudnnCreate(handle));
  }
};

ConvolutionLayer::ConvolutionLayer(const std::vector<int>& dim, int fCount, int fSize) : filterSize(fCount){
  
  std::vector<int> dimensions(1, fCount);
  for(int i = 0; i < dim.size(); i++){
    if(i+2 < dim.size()){ //last two dimensions give the frame
      dimensions.push_back(dim[i]);
    }
    else{
      dimensions.push_back(fSize);
    }
  }

  filters = Tensor(dimensions);
  bias = Tensor({1, filters.size});
  CudaM = CudaMembers();
}

void ConvolutionLayer::forward(Tensor T){
  
}

