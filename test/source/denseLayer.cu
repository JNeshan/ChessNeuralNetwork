#include "../header/denseLayer.h"
#include <cuda_runtime.h>
#include <cublas.h>
#include <stdexcept>

//should be done

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



__global__ void biasAddKernel(const float* bias, float* out, const int n, const int m){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n * m){
    out[idx] += bias[idx % n];
  }
}

__global__ void bGradKernel(const float* grad, float* out, const int m, const int n){
  extern __shared__ float shared[]; //initializes space for shared memory
  const int colIdx = blockIdx.x; //column index the thread block works on
  const int thId = threadIdx.x; //threads relative id to its block
  shared[thId] = 0.0f; //initializing shared memory values
  for(int row = thId; row < m; row += blockDim.x){ //adding gradient values to shared memory
    shared[thId] += grad[row * n + colIdx];
  }
  __syncthreads(); //waits for all threads to finish
  //essentially starts by cutting off half the active threads, only the lower half index of each block starts up to thread th0, 
  //then each active thread adds the value k indices away from their own, where k is 1 plus the index of th0.
  //This then continues to merge down each column until the final two indices containing the sums of their halfs of the vector merge for that
  //rows gradient
  for(int str = blockDim.x / 2; str > 0; str >>= 1){ //summing each column, applying half to the other half each time
    if(thId < str){ //indicates which threads are still allowed
      shared[thId] += shared[thId + str];
    }
    __syncthreads(); //waits to sync each iteration
  }

  if(thId == 0){ //the last thread sets the columns value
    out[colIdx] = shared[0]; 
  }
}

DenseLayer::DenseLayer(const int f, const int n) : weight({f, n}, TensorLocation::GPU), bias({n}, TensorLocation::GPU), wGrad({f, n}, TensorLocation::GPU), bGrad({n}, TensorLocation::GPU), input(){}

Tensor DenseLayer::forward(const Tensor& T){
  if(T.n != 2){
    throw("Wrong dimensional tensor for dense layer.");
  }
  if(T.dimensions[1] != weight.dimensions[0]){
    throw("Weight and input tensor dimensions incompatible for multiplication");
  }
  
  input = Tensor(T);

  Tensor output({T.dimensions[0], weight.dimensions[1]}, TensorLocation::GPU);
  TryCuda(cublasSgemm_v2(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, weight.dimensions[1], T.dimensions[0], T.dimensions[1],
             &mx, weight.gpuData(), weight.dimensions[1], T.gpuData(), T.dimensions[1], &mn, output.gpuData(), output.dimensions[1]));
  const int thCount = 256, m = (output.size + thCount - 1) / thCount;
  dim3 blockDim(thCount);
  dim3 gridDim(m);
  biasAddKernel<<<gridDim, blockDim>>>(bias.gpuData(), output.gpuData(), output.dimensions[0], output.dimensions[1]);
  return output;
}

Tensor DenseLayer::backward(const Tensor& gradient){
  
  Tensor iGrad(input.dimensions, TensorLocation::GPU, input.n);
  //calculates the input gradient
  TryCuda(cublasSgemm_v2(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, gradient.dimensions[0], weight.dimensions[0], 
                        gradient.dimensions[1], &mx, gradient.gpuData(), gradient.dimensions[1], weight.gpuData(), 
                        weight.dimensions[1], &mn, iGrad.gpuData(), iGrad.dimensions[1]));
  //calculates weight gradient 
  TryCuda(cublasSgemm_v2(blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, input.dimensions[1], gradient.dimensions[1], 
                        input.dimensions[0], &mx, input.gpuData(), input.dimensions[1], gradient.gpuData(), 
                        gradient.dimensions[1], &mn, wGrad.gpuData(), wGrad.dimensions[1]));
  
  const int thCount = 256; //threads per block
  dim3 gridDim(gradient.dimensions[1]);
  dim3 blockDim(thCount); //one dimensional block of th threads
  size_t shrMemSize = thCount * sizeof(float); //size in memory of each block
  bGradKernel<<<gridDim, blockDim, shrMemSize>>>(gradient.gpuData(), bGrad.gpuData(), gradient.dimensions[0], gradient.dimensions[1]);
  return iGrad;
}

void DenseLayer::genTensorData(){
  Generator::tGen(bias);
  Generator::tGen(weight);
}

void DenseLayer::loadTensor(std::ifstream& iF){
  if(!iF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }

  weight.readBinary(iF);
  bias.readBinary(iF);
  weight.gpuSend();
  bias.gpuSend();
}

void DenseLayer::saveTensor(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }

  weight.writeBinary(oF);
  bias.writeBinary(oF);
  weight.gpuSend();
  bias.gpuSend();
}

void DenseLayer::cleanSave(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }
  oF << "Dense Layer Tensors:\nWeights ";
  weight.writeTensor(oF);
  oF << "Bias ";
  bias.writeTensor(oF);
  weight.gpuSend();
  bias.gpuSend();
}

DenseLayer::~DenseLayer(){}