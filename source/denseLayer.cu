#include "header/convolutionLayer.h"
#include <cuda_runtime.h>
#include <cublas.h>
#include <stdexcept>

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}



struct CudaMembers{
  cublasStatus_t stat;

};