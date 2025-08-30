#include "../header/generator.h"

__inline__ void TryCuda(curandStatus_t err){
  if(err != CURAND_STATUS_SUCCESS){
    fprintf(stderr, "curand Error in %s at line %d: %s\n", __FILE__, __LINE__, (char)('0' + err));
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "curand Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ curandGenerator_t createCurand(){
  curandGenerator_t gen;
  TryCuda(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  TryCuda(curandSetPseudoRandomGeneratorSeed(gen, 0));
  TryCuda(curandSetGeneratorOffset(gen, 1));
  TryCuda(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT));
  return gen;
}

__global__ void ascendKernel(const int s, float* data){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < s){
    data[idx] = idx;
  }
}

__global__ void valKernel(const int s, const int v, float* data){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < s){
    data[idx] = v;
  }
}

__global__ void copyKernel(const int s, const float* r, float* data){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < s){
    data[idx] = r[idx];
  }
}



curandGenerator_t Generator::cGen = createCurand();

Generator::Generator(){
  int a = 0;
}

void Generator::tGen(const Tensor& T){
  float* output = T.gpuData();
  TryCuda(curandGenerateUniform(cGen, output, T.size));
  return;
}

void Generator::dGen(const int s, float* data){
  TryCuda(curandGenerateUniform(cGen, data, s));
}

void Generator::aGen(const int s, float* data){
  const int thCount = 256, m = (s + thCount - 1) / thCount;
  dim3 blockDim(thCount);
  dim3 gridDim(m);
  ascendKernel<<<gridDim, blockDim>>>(s, data);
  TryCuda(cudaGetLastError());
  TryCuda(cudaDeviceSynchronize());
}

void Generator::vGen(const int s, const int v, float* data){
  const int thCount = 256, m = (s + thCount - 1) / thCount;
  dim3 blockDim(thCount);
  dim3 gridDim(m);
  valKernel<<<gridDim, blockDim>>>(s, v, data);
  TryCuda(cudaGetLastError());
  TryCuda(cudaDeviceSynchronize());
}

void Generator::copy(const int s, const float* r, float* data){
  const int thCount = 256, m = (s + thCount - 1) / thCount;
  dim3 blockDim(thCount);
  dim3 gridDim(m);
  copyKernel<<<gridDim, blockDim>>>(s, r, data);
  TryCuda(cudaGetLastError());
  TryCuda(cudaDeviceSynchronize());
}