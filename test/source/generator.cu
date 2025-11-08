#include "../header/generator.h"

__inline__ const char* curandGetErrorString(curandStatus_t error){
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
    default:
      return "CURAND_STATUS_UNKNOWN_ERROR";
  }
}

__inline__ void TryCuda(curandStatus_t err) {
  if(err != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRAND Error in %s at line %d: %s (code %d)\n", __FILE__, __LINE__, curandGetErrorString(err), err);
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCuda(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "Cuda Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
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



thread_local curandGenerator_t Generator::cGen = createCurand();

Generator::Generator(){
  int a = 0;
}

void Generator::tGen(Tensor<float>& T){
  float* output = T.gpuData();
  TryCuda(curandGenerateUniform(cGen, output, T.size));
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