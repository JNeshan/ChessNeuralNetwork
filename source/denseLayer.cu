#include "header/denseLayer.h"
#include <cuda_runtime.h>
#include <cublas.h>
#include <cudnn.h>
#include <stdexcept>

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCuda(cublasStatus_t err){
  if(err != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "cuBLAS Error in %s at line %d: %s\n", __FILE__, __LINE__, cublasGetStatusString(err));
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
  cublasHandle_t handle;
  cudnnHandle_t nHandle;
  cudnnTensorDescriptor_t outputD, biasD, gradientD;

  CudaMembers(){
    cublasCreate_v2(&handle);
    cudnnCreate(&nHandle);
    cudnnCreateTensorDescriptor(&outputD);
    cudnnCreateTensorDescriptor(&biasD);
    TryCuda(cudnnCreateTensorDescriptor(&gradientD));

  }

  ~CudaMembers(){
    TryCuda(cublasDestroy_v2(handle));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnDestroyTensorDescriptor(biasD));
    TryCuda(cudnnDestroy(nHandle));
    TryCuda(cudnnDestroyTensorDescriptor(gradientD));
  };

  void resetTemp(){
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
  }
};
//uses
__global__ void bGradKernel(const float* grad, float* out, const int m, const int n){
  extern __shared__ float shared[]; //initializes space for shared memory
  const int colIdx = blockIdx.x; //column index the thread block works on
  int thId = threadIdx.x; //threads relative id
  shared[thId] = 0.0f; //initializing shared memory values
  for(int row = thId; row < m; row++){ //adding gradient values to shared memory
    shared[thId] += grad[row * n + colIdx];
  }
  __syncthreads();

  for(int str = blockDim.x / 2; str > 0; str >>= 1){ //summing each column, applying half to the other half each time
    if(thId < str){ //indicates which threads are still allowed
      shared[thId] += shared[thId + str];
    }
    __syncthreads(); //waits to sync each iteration
  }

  if(thId == 0){ //one thread sets the sum of its column
    out[colIdx] = shared[0]; 
  }
}

DenseLayer::DenseLayer(const int f, const int n) : weight({f, n}, TensorLocation::GPU), bias({1, n}, TensorLocation::GPU){
  CudaM = new CudaMembers();
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, 1, 1, 1));
}


Tensor DenseLayer::forward(const Tensor& T){
  if(T.dimensions[1] != weight.dimensions[0]){
    throw("Weight and input tensor dimensions incompatible for multiplication");
  }
  input = T;
  Tensor output({T.dimensions[0], weight.dimensions[1]}, TensorLocation::GPU);
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.dimensions[0], output.dimensions[1], 1, 1));
  TryCuda(cublasSgemm_v2(CudaM->handle, CUBLAS_OP_N, CUBLAS_OP_N, weight.dimensions[1], input.dimensions[0], input.dimensions[1],
                       &mx, weight.gpuData(), weight.dimensions[1], input.gpuData(), input.dimensions[1], &mn, output.gpuData(), output.dimensions[2]));
  TryCuda(cudnnAddTensor(CudaM->nHandle, &mx, CudaM->outputD, output.gpuData(), &mx, CudaM->biasD, bias.gpuData()));
  return output;
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> DenseLayer::backward(const Tensor& gradient){
  iGrad = Tensor (input.dimensions, TensorLocation::GPU);
  wGrad = Tensor (weight.dimensions, TensorLocation::GPU);
  bGrad = Tensor (bias.dimensions, TensorLocation::GPU);

  TryCuda(cublasSgemm_v2(CudaM->handle, CUBLAS_OP_T, CUBLAS_OP_N, gradient.dimensions[0], weight.dimensions[0], 
                        gradient.dimensions[1], &mx, gradient.gpuData(), gradient.dimensions[1], weight.gpuData(), 
                        weight.dimensions[1], &mn, iGrad.gpuData(), iGrad.dimensions[1]));
  //calculates weight gradient by 
  TryCuda(cublasSgemm_v2(CudaM->handle, CUBLAS_OP_N, CUBLAS_OP_T, input.dimensions[1], gradient.dimensions[1], 
                        input.dimensions[0], &mx, input.gpuData(), input.dimensions[1], gradient.gpuData(), 
                        gradient.dimensions[1], &mn, wGrad.gpuData(), wGrad.dimensions[1]));
  //TryCuda()
  int thCount = 256; //threads per block
  while(thCount < input.dimensions[0]){
    thCount *= 2;
  }
  dim3 gridDim(gradient.dimensions[1]);
  dim3 blockDim(thCount); //one dimensional block of th threads
  size_t shrMemSize = thCount * sizeof(float); //size in memory of each block
  bGradKernel<<<gridDim, blockDim, shrMemSize>>>(gradient.gpuData(), bGrad.gpuData(), gradient.dimensions[0], gradient.dimensions[1]);
  CudaM->resetTemp();
  return {{&input, &weight, &bias}, {&iGrad, &wGrad, &bGrad}};
}

DenseLayer::~DenseLayer(){
  delete CudaM;
}