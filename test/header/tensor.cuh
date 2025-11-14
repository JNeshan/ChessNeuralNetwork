//tensor.cuh
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <atomic>
#include <fstream>
#include <memory>
#include <map>
#include <cuda_fp16.h>
#include <type_traits>
#include <string>
#include <sstream>
#include <iomanip>
#include "cudnn.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

enum class TensorLocation {CPU, GPU}; //denotes the CPU/host and GPU/device
enum class AllocationType {FIXED, DYNAMIC}; //denotes whether a tensor uses standard blocking cuda mem calls or aync stream-based cuda mem calls

//The tensor class wraps a pointer to a contiguous array with data to represent it as an n-dimensional matrix, along with
//handling the allocation, destruction, and transfer of memory in the CPU and GPU
//The dimensions and overall size are necessary to perform the Cuda computations, and the dimension count n is needed to
//enforce constraints for a layer that expects an input tensor with specific dimensions (i.e. convolution using 4, dense using 2)
//Reshape can freely modify dimensions and n as long as size remains the same, since its encapsulating the data but with a different
//split.
//NOTE: Tensor will always pad dimensions to be at least size 4. This is essential since cudnn tensor descriptors require 4
//dimension inputs, so by padding dimensions with 1, which keeps size the same, and using n to recall the actual dimensionality,
//the descriptor calls can always use the four indices of dimensions. This just makes it much simpler.
//
//Tensors have their data including their sizes, dimensionality, and array data stored into binary files. They are written to the immediately
//open section so sequential calls to write to the same open file will write them sequentially. As such the read function expects to read
//tensor information from an input file in a very strict order, with it using the data type sizes to move across the data and information like
//size to know how far to extend the float*. As such, the order that the tensors are saved in needs to be maintained across runs, so for the
//overall neural network, a modification to the layers in any form means that the data either needs to be reset, or there needs to be a function
//specifically made to change the structure of layers and maintain the old layers data.
//The main draw from this is that writing and reading must be performed very carefully.
//
//



/*
Templated class modifications
Change to allow either half precision or full precision float using cuda_fp16 and using NHWC instead of NCHW
Descriptors rely on format description so standardizing is necessary for refactor

The following must be true to allow this class to function
A tensor will only be up to 4 dimensional and must have real dimensions
Dimensions is always in NHWC format with 4 elements
Non-existent dimensions on lesser-dimensional tensors have a value of 1
A half tensor will never need to move its data to the cpu
A tensor of one type cannot perform operations meant for the other type
T is either a half or a float
If T is a half, the tensor will always be in the GPU
If T is a float, the tensor can move between the CPU and GPU
A tensor can be freely reshaped as long as its total size and elements are unchanged
Two tensors can be added as long as they have the same size and T data type
A tensor with -1 size is a blank object, is marked as stored in the GPU, should only be used to perform batch build function
*/


__inline__ const char* curandGetErrorStringTen(curandStatus_t error) {
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

__inline__ void TryCudaTen(curandStatus_t err){
  if(err != CURAND_STATUS_SUCCESS){
    fprintf(stderr, "cuRAND Error in %s at line %d: %s (code %d)\n", __FILE__, __LINE__, curandGetErrorStringTen(err), err); 
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCudaTen(cudaError_t err){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCudaTen(cudnnStatus_t err){
  if(err != CUDNN_STATUS_SUCCESS){
    fprintf(stderr, "CUDNN Error in %s at line %d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__inline__ void TryCudaTen(cublasStatus_t err){
  if(err != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "cuBLAS Error in %s at line %d: %s\n", __FILE__, __LINE__, cublasGetStatusString(err));
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

//converts every element in src from a cuda half to a float and stores in dst
static __global__ void HalfToFullKernel(float* dst, const __half* src, const int size){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x; //threads relative position in the array based on ids, dimensions, and blocks; that's just really cool
  if(idx < size){
    dst[idx] = __half2float(src[idx]); //converting to float
  }
}

//converts every element in src from a float to a cuda half and stores in dst
static __global__ void FullToHalfKernel(__half* dst, const float* src, const int size){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size){
    dst[idx] = __float2half(src[idx]);  //converting to half
  }
}

//performs element-wise addition on A and B and writes the results to out
template<typename T, typename U, typename V>
__global__ void AddKernel(const T* A, const U* B, V* out, const int size){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size) return;
  if constexpr(std::is_same_v<V, __half>){
    float sum = __half2float(A[idx]) + __half2float(B[idx]); //will implicitly convert any half to float - demonstratably incorrect
    out[idx] = __float2half_ru(sum); //converts back to half
  }
  else{
    out[idx] = __half2float(A[idx]) + __half2float(B[idx]); //implicitly converts so it works
  }
}

struct StreamDeleter{
  void operator()(cudaStream_t* p) const noexcept{
    if(p){
      TryCudaTen(cudaStreamDestroy(*p));
    }
  }
};

template <typename T, AllocationType ALLOC>
struct HostDeleter{
  HostDeleter() = default;
  void operator()(T* p) const noexcept{
    if(p){
      if(ALLOC == AllocationType::FIXED){
        delete[] p;
      }
      else{
        TryCudaTen(cudaFreeHost(p));
      }
    }
  }
};

//custom deleter for unique_ptr<T> in GPU memory (CUDA)
template <typename T, AllocationType ALLOC>
struct CudaDeleter{
  std::shared_ptr<cudaStream_t> stream = nullptr;
  CudaDeleter() = default;
  CudaDeleter(std::shared_ptr<cudaStream_t>& str) : stream(str){
  }
  void operator()(T* p) const noexcept{ //called whenever the object owned by unique_ptr<T, CudaDeleter> is deleted with the pointer to the T value contianed passed as p
    if(p){
      if(ALLOC == AllocationType::FIXED){ 
        TryCudaTen(cudaFree(p));
      }
      else{
        TryCudaTen(cudaFreeAsync(p, *this->stream));
      }
    }
  }
};

//custom key struct for tensor descriptor map
template <typename T, AllocationType ALLOC>
class Tensor;

struct TensorKey{
  std::vector<int> dimensions;
  int n = 0;
  int dataType = 0; //represents a tensors templated T type

  template <typename T, AllocationType ALLOC>
  TensorKey(const Tensor<T, ALLOC>& ten){
    this->dimensions = ten.dimensions;
    this->n = ten.n;
    if constexpr(std::is_same_v<T, float>){
      dataType = 2;
    }
    else if constexpr(std::is_same_v<T, __half>){
      dataType = 1;
    }
    else{ //int8_t
      dataType = 0;
    }
  }

  
  //tensor key values considered less than for the tensor with the first lower value in order of priority
  //from dataType (float > __half > int8_t) > n (dimensionality) > dimensions[0] > ... > dimensions[3] 
  //this ensures that if two tensors share the same traits used by a descriptor they will use the same descriptor
  bool operator<(const TensorKey& rK) const noexcept{
    return std::tie(this->dataType, this->n, this->dimensions[0], this->dimensions[3], this->dimensions[1], this->dimensions[2])
    < std::tie(rK.dataType, rK.n, rK.dimensions[0], rK.dimensions[3], rK.dimensions[1], rK.dimensions[2]); //channels more likely to differ
  }
};



template <typename T = float, AllocationType ALLOC = AllocationType::FIXED> 
class Tensor {
public:

  inline static std::map<TensorKey, cudnnTensorDescriptor_t> descriptors{};
  inline static std::atomic<size_t> totalDeviceMemAllocation{0};
  inline static std::atomic<size_t> totalHostMemAllocation{0};

  static_assert(std::is_floating_point_v<T> || std::is_same_v<T,__half> || std::is_same_v<T,int8_t>, "Tensor<T, ALLOC>: T must be a floating point type or half "); //enforces data type restriction

  //destructor
  ~Tensor(){
    this->deallocate();
  } 

  //blank constructor with light molding support
  Tensor(TensorLocation loc = TensorLocation::GPU, cudaStream_t* str = nullptr) : stream(str, StreamDeleter()), hostData(nullptr, HostDeleter<T, ALLOC>()), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)){
    this->device = loc;
    this->n = -1;
    this->size = -1;
    this->dimensions = {};
  }

  //standard tensor constructor
  Tensor(const std::vector<int>& dim, const TensorLocation loc = TensorLocation::CPU, int nth = 4, cudaStream_t* str = nullptr) : dimensions(dim), device(loc), n(nth), stream(str, StreamDeleter()), hostData(nullptr, HostDeleter<T, ALLOC>()), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)){
    if constexpr(std::is_same_v<T, __half>){ //remove if cpu half added
      if(loc == TensorLocation::CPU){
        throw std::runtime_error("Half tensor being constructed on CPU");
      }
    } 

    if(dim.size() != 4){ //dimensions shou
      throw std::runtime_error("Given dimensions does not have 4 elements");
    }
    if(nth <= 0 || nth > 4){
      throw std::runtime_error("Invalid nth in constructor");
    }
    this->size = 1;
    for(auto x : dim){ //computes size and ensures valid dimensions
      if(x <= 0){
        throw std::runtime_error("Non-positive dimension given");
      }
      this->size *= x;
    }
    this->allocate();
  }
  //overload for easy 2d construction
  Tensor(int batches, int inputChannels, const TensorLocation loc = TensorLocation::CPU, cudaStream_t* str = nullptr) : Tensor({batches, 1, 1, inputChannels}, loc, 2, str){}
  //overload for easy 1d construction
  Tensor(int inputChannels, const TensorLocation loc = TensorLocation::CPU, cudaStream_t* str = nullptr) : Tensor({1, 1, 1, inputChannels}, loc, 1, str){}
  //constructor that models after another tensor and allocates space to the specified device without copying the data
  Tensor(const Tensor<T, ALLOC>& r, const TensorLocation loc) : stream(r.stream), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)), hostData(nullptr, HostDeleter<T, ALLOC>()){
    this->dimensions = r.dimensions;
    this->n = r.n;
    this->size = r.size;
    this->device = loc;
    this->allocate();
  }

  //default copy constructor
  Tensor(const Tensor<T, ALLOC>& r) : stream(r.stream), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    this->copy(r);
  }

  template<typename U, AllocationType A>
  Tensor(const Tensor<U, A>& r) : stream(r.stream), hostData(nullptr, HostDeleter<T, ALLOC>()), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    this->copy(r);
  }

  //move constructors
  //untemplated
  Tensor(Tensor<T, ALLOC>&& r) : stream(r.stream), hostData(nullptr, HostDeleter<T, ALLOC>()), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)){
    this->size = r.size;
    this->device = r.device;
    this->dimensions = r.dimensions;
    this->n = r.n;
    this->hostData.swap(r.hostData);
    this->deviceData.swap(r.deviceData);
  }

  //templated move constructor
  //default will override when possible so
  //this is only used for type conversion
  //which requires a full copy anyways
  template <typename U>
  Tensor(Tensor<U, ALLOC>&& r) : stream(r.stream), hostData(nullptr, HostDeleter<T, ALLOC>()), deviceData(nullptr, CudaDeleter<T, ALLOC>(this->stream)){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    this->copy(r);
  }
  
  //move assignment operators
  
  //default move assignment operator
  Tensor<T, ALLOC>& operator=(Tensor<T, ALLOC>&& r){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    if(this->device == r.device){
      this->size = r.size;
      this->device = r.device;
      this->dimensions = r.dimensions;
      this->n = r.n;
      this->hostData.swap(r.hostData); 
      this->deviceData.swap(r.deviceData);
    }
    else{
      this->copy(r); //cross device transfer is as fast as deep copy, unless implemeneted page locking
    }
    return *this;
  }
  //move assignment operator
  template <typename U>
  Tensor<T, ALLOC>& operator=(Tensor<U, ALLOC>&& r){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    this->copy(r);
    return *this;
  }

  //default assignment operator
  Tensor<T, ALLOC>& operator=(const Tensor<T, ALLOC>& r){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    this->copy(r);
    return *this;
  }

  template <typename U, AllocationType A>
  Tensor<T, ALLOC>& operator=(const Tensor<U, A>& r){
    if(r.size <= 0){
      throw std::runtime_error("Attemping to copy blank tensor");
    }
    this->copy(r);
    return *this;
  }
  //conversion operator overload for TensorKey
  operator TensorKey() const {
    return TensorKey(*this);
  }
  //currently defunct, replaced by operator overloads
  void gpuAdd(Tensor<T, ALLOC>& B){
    if(this->size != B.size){
      throw std::runtime_error("Different sizes for addition");
    }
    if(this->size <= 0){
      throw std::runtime_error("Unpopulated tensor passed to operator overloard");
    }

    this->gpuSend(); //ensuring data is on GPU
    B.gpuSend();
    
    auto* aData = this->gpuData(); //retrieve data references
    auto* bData = B.gpuData();
    //kernel instantiation variables
    int thrdCnt = 256; //threads per thread block
    dim3 gridDim((this->size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
    AddKernel<<<gridDim, blockDim, 0, *this->stream>>>(aData, bData, aData, this->size);
    TryCudaTen(cudaDeviceSynchronize()); //templated kernel call

  }

  //addition operator overload performs element-wise addition over the tensor matrices
  Tensor<T, ALLOC> operator+(const Tensor<T, ALLOC>& rT) const{
    if(this->size != rT.size){
      throw std::runtime_error("Addition operator overload recieved incompatible tensors");
    }
    if(this->device != rT.device){
      throw std::runtime_error("Addition operator overload recieved tensors on seperate memory");
    }
    if(this->size <= 0){
      throw std::runtime_error("Unpopulated tensor passed to operator overloard");
    }
    Tensor<T, ALLOC> out(*this, this->device);
    if(this->device == TensorLocation::CPU){
      if constexpr(!std::is_floating_point_v<T>){
        throw std::runtime_error("Half stored on CPU in addition overload"); //fix
      }
      auto* aData = this->hostData.get();
      auto* bData = rT.hostData.get();
      auto* outPtr = out.hostData.get();
      for(int i = 0; i < this->size; i++){
        outPtr[i] = aData[i] + bData[i];
      }
    }
    else{
      auto* aData = this->deviceData.get();
      auto* bData = rT.gpuData();
      auto* outPtr = out.deviceData.get();
      //kernel instantiation variables
      int thrdCnt = 256; //threads per thread block
      dim3 gridDim((this->size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
      AddKernel<<<gridDim, blockDim, 0, *this->stream>>>(aData, bData, outPtr, this->size);
      //TryCudaTen(cudaDeviceSynchronize()); //templated kernel call
    }
    return out;
  }
  //overloads += operator to perform element wise addition, tensors must have equal size
  
  Tensor<T, ALLOC>& operator+=(const Tensor<T, ALLOC>& rT){
    if(this->size != rT.size){
      throw std::runtime_error("Different sizes for addition assignment overload");
    }
    if(this->device != rT.device){
      throw std::runtime_error("Addition assignment operator overload recieved tensors on seperate memory");
    }
    if(this->size <= 0){
      throw std::runtime_error("Unpopulated tensor passed to operator overloard");
    }
    if(this->device == TensorLocation::CPU){
      if constexpr(!std::is_floating_point_v<T>){ //clears out half check for CPU
        throw std::runtime_error("Half stored on CPU in addition assignment overload");
      }
      auto* aData = this->hostData.get(); //reference pointers
      auto* bData = rT.hostData.get();
      for(int i = 0; i < this->size; i++){ //performs element-wise addition
        aData[i] += bData[i];
      }
    }
    else{
      auto* aData = this->deviceData.get(); //reference pointers
      auto* bData = rT.gpuData();
      //kernel instantiation variables
      int thrdCnt = 256; //threads per thread block
      dim3 gridDim((this->size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
      AddKernel<<<gridDim, blockDim, 0, *this->stream>>>(aData, bData, aData, this->size);
      //TryCudaTen(cudaDeviceSynchronize()); //templated kernel call
    }
    return *this;
  }

  template <typename U>
  Tensor<T, ALLOC>& operator+=(const Tensor<U, ALLOC>& rT){
    if(this->size != rT.size){
      throw std::runtime_error("Different sizes for addition assignment overload");
    }
    if(this->device != rT.device){
      throw std::runtime_error("Addition assignment operator overload recieved tensors on seperate memory");
    }
    if(this->size <= 0){
      throw std::runtime_error("Unpopulated tensor passed to operator overloard");
    }

    if constexpr(std::is_same_v<T, U>){
      if(this->device == TensorLocation::CPU){
        if constexpr(!std::is_floating_point_v<T>){ //clears out half check for CPU
          throw std::runtime_error("Half stored on CPU in addition assignment overload");
        }
        auto* aData = this->hostData.get(); //reference pointers
        auto* bData = rT.hostData.get();
        for(int i = 0; i < this->size; i++){ //performs element-wise addition
          aData[i] += bData[i];
        }
      }
      else{
        auto* aData = this->deviceData.get(); //reference pointers
        auto* bData = rT.gpuData();
        //kernel instantiation variables
        int thrdCnt = 256; //threads per thread block
        dim3 gridDim((this->size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
        AddKernel<<<gridDim, blockDim, 0, *this->stream>>>(aData, bData, aData, this->size);
      }
    }
    else{
        if(this->device == TensorLocation::CPU){
          throw std::runtime_error("Half stored on CPU in addition assignment overload");
        }
        auto* aData = this->deviceData.get(); //reference pointers
        auto* bData = rT.gpuData();
        //kernel instantiation variables
        int thrdCnt = 256; //threads per thread block
        dim3 gridDim((this->size + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
        AddKernel<<<gridDim, blockDim, 0, *this->stream>>>(aData, bData, aData, this->size);
    }
    return *this;
  }

  //takes set of tensors and copies them into one larger tensor on page-locked CPU memory
  //requires the tensor was set as PageLocked and is currently set to the CPU memory
  //the gpu can read and write to the batch very quickly 
  void batchBuild(const std::vector<Tensor<T, ALLOC>>& tensorBatch){
    if(this->device == TensorLocation::GPU){
      throw std::runtime_error("Main batching tensor must be in CPU memory");
    }
    if(tensorBatch.empty()){
      throw std::runtime_error("Set of passed tensors to batch empty");
    }

    if(this->dimensions[0] != tensorBatch.size()){ //skip if batch tensor is already configured for the batch
      this->deallocate(); //resets to get greater size
      this->n = 4; //setting basic members
      this->dimensions = tensorBatch[0].dimensions;
      this->dimensions[0] = tensorBatch.size();
      this->size = tensorBatch[0].size;
      this->size *= tensorBatch.size();
      this->allocate();
    }
    
    const int gap = tensorBatch[0].size; //the number of elements per tensor being batched
    int offset = 0; //"this" ptr offset value
    T* refPtr = this->hostData.get();
    for(auto& tRef : tensorBatch){
      if(tRef.size != gap){
        throw std::runtime_error("Tensor batch contains differently sized tensors");
      }
      if(tRef.device != TensorLocation::CPU){
        throw std::runtime_error("All tensors in batch build must be on the CPU");
      }
      memcpy(refPtr + offset, tRef.hostData.get(), gap * sizeof(T));
      offset += gap; //incrememnts the offset by however many elements were added
    }
    
  }
  //takes a section of the tensor from the batches at the specified index
  //rework to use a special struct that holds a view and a shared ptr and can be converted back into a tensor
  Tensor<T, ALLOC> segment(const int n_i){
    if(n_i < 0 || n_i >= this->dimensions[0]){
      throw std::runtime_error("Bad batch index to copy");
    }
    std::vector<int> segDim = this->dimensions; //each tensor in a batch has the same dimensions with the dimensions[0] always being a tensors batch size
    segDim[0] = 1;
    Tensor<T, ALLOC> outTensor(segDim, this->device, this->n, this->stream.get());
    int offset = n_i * outTensor.size; //marks offset of desired tensor
    if(this->device == TensorLocation::CPU){
      if constexpr(!std::is_floating_point_v<T>){
        throw std::runtime_error("Half stored on CPU in segment");
      }
      memcpy(outTensor.hostData.get(), this->hostData.get() + offset, outTensor.size * sizeof(T));
    }
    else if constexpr(ALLOC == AllocationType::FIXED){
      TryCudaTen(cudaMemcpy(outTensor.gpuData(), this->deviceData.get() + offset, outTensor.size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    else{
      TryCudaTen(cudaMemcpyAsync(outTensor.gpuData(), this->deviceData.get() + offset, outTensor.size * sizeof(T), cudaMemcpyDeviceToDevice, *this->stream));
    }
    return outTensor;
  }
  //returns the value stored at the specified set of indices
  T at(std::vector<int> ind){
    int indice = 0, mult = 1;
    for(int i = ind.size() - 1; i >= 0; i--){
      if(ind[i] >= this->dimensions[i] || ind[i] < 0){ //ensuring indices are all positive in bound values
        throw std::out_of_range("Indice array contains invalid index value");
      }
      indice += ind[i] * mult;
      mult *= this->dimensions[i];
    }
    if(this->device == TensorLocation::CPU){
      return this->hostData.get()[indice];
    }
    else{
      T val;
      TryCudaTen(cudaMemcpy(&val, this->deviceData.get() + indice, sizeof(T), cudaMemcpyDeviceToHost));
      return val;
    }

  }
  //returns the value stored at the specified 4 dimensional index
  T at(int n, int h, int w, int c){
    if(this->n != 4){
      throw std::runtime_error("Non 4-dimensional tensor tried 4-dimensional indexing");
    }
    return this->at({n, h, w, c});
  }
  //returns the value stored at the specified 2 dimensional index
  T at(int n, int c){
    if(this->n != 2){
      throw std::runtime_error("Non 2-dimensional tensor tried 2-dimensional indexing");
    }
    return this->at({n, 1, 1, c});
  }
  //returns the value stored at the specified 1 dimensional index
  T at(int i){
    if(i >= this->size || i < 0){
      throw std::out_of_range("Invalid single index given");
    }
    if(this->device == TensorLocation::CPU){
      return this->hostData.get()[i];
    }
    else{
      T val;
      TryCudaTen(cudaMemcpy(&val, this->deviceData.get() + i, sizeof(T), cudaMemcpyDeviceToHost));
      return val;
    }
  }
  
  //takes a pointer to a tensor descriptor and sets it based on the calling tensors data
  void genDescriptor(cudnnTensorDescriptor_t* dsc){
    if(this->size <= 0){
      throw std::runtime_error("Empty tensor generating descriptor");
    }

    TensorKey key(*this);
    if(descriptors.find(key) == descriptors.end()){ //if a descriptor for this specific tensor config hasn't been made one is and is cached in the map
      descriptors[key]; //initializing value at key index safely
      cudnnDataType_t dataType;
      if constexpr(std::is_same_v<T, float>){
        dataType = CUDNN_DATA_FLOAT;
      }
      else if constexpr(std::is_same_v<T, __half>){
        dataType = CUDNN_DATA_HALF;
      }
      else{
        dataType = CUDNN_DATA_INT8;
      }
      TryCudaTen(cudnnCreateTensorDescriptor(&descriptors[key]));
      TryCudaTen(cudnnSetTensor4dDescriptor(descriptors[key], CUDNN_TENSOR_NHWC, dataType, this->dimensions[0], this->dimensions[3], this->dimensions[1], this->dimensions[2]));
    }
    dsc = &descriptors[key]; //sets to reference appropriate descriptor
  }

  //modifies the shape of the matrix to a new set of dimensions that must be the same size as the previous, only modifies dimensions and n
  void reshape(const std::vector<int>& dim, const int nth){
    if(dim.size() != 4 || nth <= 0 || nth > 4){
      throw std::runtime_error("Invalid reshape dimension values passed");
    }
    int sz = 1;
    for(auto x : dim){
      if(x <= 0){
        throw std::runtime_error("Non-positive dimension passed on reshape");
      }
      sz *= x;
    }
    if(sz != this->size){ //reshape must maintain size
      throw std::runtime_error("Different reshape size");
    }
    this->dimensions = dim;
    this->n = nth;
  }
  //sends data to cpu from gpu, only usable when T is float
  template <typename U = T>
  std::enable_if_t<std::is_floating_point_v<U>, void> //enabled if templated type is float
  cpuSend(){
    if(this->device == TensorLocation::CPU){ //skips
      std::cout<<"Data already stored on CPU"<<std::endl;
      return;
    }
    if(this->deviceData == nullptr){
      throw std::runtime_error("Sending unallocated data");
    }
    
    if constexpr(ALLOC == AllocationType::FIXED){
      this->hostData.reset(new T[this->size]);
      TryCudaTen(cudaMemcpy(this->hostData.get(), this->deviceData.get(), this->getMemUsage(), cudaMemcpyDeviceToHost));
      this->deviceData = nullptr;
      totalDeviceMemAllocation.fetch_sub(this->getMemUsage(), std::memory_order_relaxed);
    }
    else{
      if(this->hostData == nullptr){ //consider leaving dynamic tensors cpu memory allocated, but this would probably require setting up memory pooling
        T* tmpPtr = nullptr;
        TryCudaTen(cudaHostAlloc((void**)&tmpPtr, this->getMemUsage(), 0));
        this->hostData.reset(tmpPtr);
      }
      TryCudaTen(cudaMemcpyAsync(this->hostData.get(), this->deviceData.get(), this->getMemUsage(), cudaMemcpyDeviceToHost, *this->stream));
      this->deviceData = nullptr;
      totalDeviceMemAllocation.fetch_sub(this->getMemUsage(), std::memory_order_relaxed);
      totalHostMemAllocation.fetch_add(this->getMemUsage(), std::memory_order_relaxed);
    }
  }

  //sends data to gpu from cpu
  template <typename U = T>
  std::enable_if_t<std::is_floating_point_v<U>, void> //enabled if templated type is float
  gpuSend(){
    if(this->device == TensorLocation::GPU){
      std::cout<<"Data already stored on GPU"<<std::endl;
      return;
    }
    if(this->hostData == nullptr){
      throw std::runtime_error("Sending unallocated data");
    }

    if constexpr(ALLOC == AllocationType::FIXED){
      T* tmpPtr = nullptr;
      TryCudaTen(cudaMalloc((void**)&tmpPtr, this->getMemUsage()));
      TryCudaTen(cudaMemcpy(tmpPtr, this->hostData.get(), this->getMemUsage(), cudaMemcpyHostToDevice));
      this->deviceData.reset(tmpPtr);
      this->hostData = nullptr;
      totalDeviceMemAllocation.fetch_add(this->getMemUsage());
    }
    else{ //consider leaving dynamic tensors cpu memory allocated, but this would probably require setting up memory pooling
      T* tmpPtr = nullptr;
      TryCudaTen(cudaMallocAsync((void**)&tmpPtr, this->getMemUsage(), *this->stream));
      TryCudaTen(cudaMemcpyAsync(tmpPtr, this->hostData.get(), this->getMemUsage(), cudaMemcpyHostToDevice, *this->stream));
      this->deviceData.reset(tmpPtr);
      this->hostData = nullptr;
      totalDeviceMemAllocation.fetch_add(this->getMemUsage());
      totalHostMemAllocation.fetch_sub(this->getMemUsage());
    }
    this->device = TensorLocation::GPU;
  }

  //returns a raw ptr to the data in cpu memory
  template <typename U = T>
  std::enable_if_t<std::is_same_v<U, float>, float*>
  cpuData() const{
    if(this->device != TensorLocation::CPU){
      throw std::runtime_error("Tensor not stored on CPU");
    }
    return this->hostData.get();
  }
  
  //returns a raw ptr to the data in gpu memory
  T* gpuData() const{
    if(this->device != TensorLocation::GPU){
      throw std::runtime_error("Tensor not stored on GPU");
    }
    return this->deviceData.get();
  }

  //file interactions for saving and restoring tensors
  //extracts tensor information from a binary input, expects the current position to start at the tensors data, extracting from files must be precise
  template <typename U = T>
  void readBinary(std::ifstream& iF){
    if(!iF.is_open()){
      std::cout<<"File not open to read"<<std::endl;
      return;
    }
    if(!std::is_floating_point_v<T>){ //has to read as float then convert to half, assignment operator handles type conversion
      Tensor<float> ten = *this;
      ten.cpuSend();
      ten.deviceData = nullptr; //clearing existing data
      ten.hostData = nullptr;
      ten.dimensions = std::vector<int>(4); 
      ten.device = TensorLocation::CPU;
      iF.read(reinterpret_cast<char*>(&ten.size), sizeof(int)); //reading data from file
      ten.hostData.reset(new T[ten.size]); //allocate with new size
      iF.read(reinterpret_cast<char*>(&ten.n), sizeof(int));
      iF.read(reinterpret_cast<char*>(ten.dimensions.data()), 4 * sizeof(int));
      iF.read(reinterpret_cast<char*>(ten.hostData.get()), ten.size * sizeof(float));
      ten.gpuSend();
      *this = ten;
    }
    else{  
      this->deviceData = nullptr; //clearing existing data
      this->hostData = nullptr;
      this->dimensions = std::vector<int>(4); 
      this->device = TensorLocation::CPU;
      iF.read(reinterpret_cast<char*>(&this->size), sizeof(int)); //reading data from file
      this->hostData.reset(new T[this->size]); //allocate with new size
      iF.read(reinterpret_cast<char*>(&this->n), sizeof(int));
      iF.read(reinterpret_cast<char*>(this->dimensions.data()), 4 * sizeof(int));
      iF.read(reinterpret_cast<char*>(this->hostData.get()), this->size * sizeof(float));
    }
  }
  //writes the tensors into binary assuming a binary output file, writes to wherever the output stream is so sequential tensor writes to the same open file can be read sequentially later
  template <typename U = T>
  void writeBinary(std::ofstream& oF){
    if(!oF.is_open()){
      std::cout<<"Output not open"<<std::endl;
      return;
    }
    if(this->size <= 0){
      std::cout<<"Tensor not allocated"<<std::endl;
      return;
    }

    if(!std::is_floating_point_v<T>){
      Tensor<float> ten = *this;
      ten.cpuSend();
      oF.write(reinterpret_cast<const char*>(&ten.size), sizeof(int)); //writing to file
      oF.write(reinterpret_cast<const char*>(&ten.n), sizeof(int));
      oF.write(reinterpret_cast<const char*>(ten.dimensions.data()), 4 * sizeof(int));
      oF.write(reinterpret_cast<const char*>(ten.hostData.get()), ten.size * sizeof(float));
      ten.gpuSend();
      *this = ten;
    }
    else{
      if(this->device == TensorLocation::GPU){
        this->cpuSend();
      }
      oF.write(reinterpret_cast<const char*>(&this->size), sizeof(int)); //writing to file
      oF.write(reinterpret_cast<const char*>(&this->n), sizeof(int));
      oF.write(reinterpret_cast<const char*>(this->dimensions.data()), 4 * sizeof(int));
      oF.write(reinterpret_cast<const char*>(this->hostData.get()), this->size * sizeof(float));
    }
    if(this->device == TensorLocation::GPU){
      this->cpuSend();
    }
  }
  //clean write for readability, converts all 2 or greater dimensional tensors to 2d for the display
  template <typename U = T>
  std::enable_if_t<std::is_floating_point_v<U>, void>
  writeTensor(std::ofstream& oF){
    int width = 16, precision = 14;
    if(!oF.is_open()){
      std::cout<<"Output stream not open"<<std::endl;
      return;
    }
    if(this->size <= 0){
      std::cout<<"Tensor not allocated"<<std::endl;
      return;
    }
    if(this->device == TensorLocation::GPU){
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
        std::string value = formatFloat(this->hostData.get()[i*y + j], width, precision);
        oF << value << " ";
      }
      oF << "\n";
    }
    oF << "\n";
  }

  //returns the size of data in bits
  size_t getMemUsage(){
    if(this->size <= 0){
      return 0;
    }
    return sizeof(T) * this->size;
  }

  //public member variables
  int size; //total number of elements and the highest order dimension
  int n; //nD tensor
  std::vector<int> dimensions; //dimension sizes
  //std::unique_ptr<T> data;
  std::unique_ptr<T[], HostDeleter<T, ALLOC>> hostData;
  std::unique_ptr<T, CudaDeleter<T, ALLOC>> deviceData;
  //enum to designate the CPU and GPU
  TensorLocation device;
  std::shared_ptr<cudaStream_t> stream; 
  //cudaStream_t* stream; //stream reference if this tensor uses dynamic allocation
  //static std::map<double, cudnnTensorDescriptor_t> descriptors;
  
  //static std::atomic<size_t> totalDeviceMemAllocation;
  //static std::atomic<size_t> totalHostMemAllocation;

private:
  //helper functions for allocating and deallocating memory space to ensure consistent tracking of size globals
  void allocate(){
    size_t s = this->size * sizeof(T);
    if(s <= 0){
      throw std::invalid_argument("Allocating with non-positive size");
    }

    this->deallocate(); //ensures proper data updates
    
    if constexpr(ALLOC == AllocationType::FIXED){
      if(this->device == TensorLocation::CPU){
        this->hostData.reset(new T[this->size]);
        //this->totalHostMemAllocation.fetch_add(s, std::memory_order_relaxed);  
      }
      else{
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMalloc((void**)&tmpPtr, s));
        this->deviceData.reset(tmpPtr);
        this->totalDeviceMemAllocation.fetch_add(s, std::memory_order_relaxed);  
      }
    }
    else{
      if(this->device == TensorLocation::CPU){
        T* tmpPtr = nullptr;
        TryCudaTen(cudaHostAlloc((void**)&tmpPtr, s, 0));
        this->hostData.reset(tmpPtr);
        this->totalHostMemAllocation.fetch_add(s, std::memory_order_relaxed);
      }
      else{
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMallocAsync((void**)&tmpPtr, s, *this->stream));
        this->deviceData.reset(tmpPtr);
        this->totalDeviceMemAllocation.fetch_add(s, std::memory_order_relaxed);
      }
    }
  }
  
  //destroys data pointers and updates global atomics, smart pointers handle proper deletion automatically
  void deallocate(){
    if(this->size <= 0){
      return;
    }
    if(this->device == TensorLocation::CPU){
      if(this->hostData == nullptr){
        return;
      }
      if constexpr(ALLOC == AllocationType::DYNAMIC){ //tracking page-locked host memory only
        this->totalHostMemAllocation.fetch_sub(this->getMemUsage(), std::memory_order_relaxed);
      }
      this->hostData = nullptr; //smart pointers automatically invoke deletion
      this->deviceData = nullptr;
    }
    else{
      if(this->deviceData == nullptr){
        return;
      }
      this->totalDeviceMemAllocation.fetch_sub(this->getMemUsage(), std::memory_order_relaxed);
      this->deviceData = nullptr;
      this->hostData = nullptr;
    }
  }
  
  //same type copy helper
  template <AllocationType A>
  void copy(const Tensor<T, A>& r){

    if(this->size != r.size){
      this->deallocate();
      this->size = r.size;
      this->n = r.n;
      this->dimensions = r.dimensions;
      this->allocate();
    }

    if(r.device == TensorLocation::CPU){ //r on cpu
      if(this->device == TensorLocation::CPU){ //both on cpu
        memcpy(this->hostData.get(), r.hostData.get(), this->getMemUsage());
      }
      else if constexpr(A == AllocationType::DYNAMIC){ //this on gpu, r dynamic tensor on cpu
        TryCudaTen(cudaMemcpyAsync(this->deviceData.get(), r.hostData.get(), this->getMemUsage(), cudaMemcpyHostToDevice, *r.stream));
      }
      else{ //this on gpu, r on cpu
        TryCudaTen(cudaMemcpy(this->deviceData.get(), r.hostData.get(), this->getMemUsage(), cudaMemcpyHostToDevice));
      }
    }
    else if constexpr(A == AllocationType::DYNAMIC){ //r dynamic tensor on gpu
      if(this->device == TensorLocation::CPU){  //this on cpu, r dynamic tensor on gpu
        TryCudaTen(cudaMemcpyAsync(this->hostData.get(), r.deviceData.get(), this->getMemUsage(), cudaMemcpyDeviceToHost, *r.stream));
      }
      else{ //this on gpu, r dynamic tensor on gpu
        TryCudaTen(cudaMemcpyAsync(this->deviceData.get(), r.deviceData.get(), this->getMemUsage(), cudaMemcpyDeviceToDevice, *r.stream));
      }
    }
    else{
      if(this->device == TensorLocation::CPU){ //this on cpu, r on gpu
        TryCudaTen(cudaMemcpy(this->hostData.get(), r.deviceData.get(), this->getMemUsage(), cudaMemcpyDeviceToHost));
      }
      else{ //this on gpu, r on gpu
        TryCudaTen(cudaMemcpy(this->deviceData.get(), r.deviceData.get(), this->getMemUsage(), cudaMemcpyDeviceToDevice));
      }
      //mfw im afraid to write monotonous comments for funsies because it will look like i used ai 8|:(
    }
  }
  //copy helper for float-half conversion, requires both devices in GPU
  //allocation type can be ignored completely
  template <typename U>
  std::enable_if_t<!std::is_same_v<T, U>, void>
  copy(const Tensor<U, ALLOC>& r){
    if(this->size != r.size){  //resets this and changes members
      this->deallocate();
      this->size = r.size;
      this->dimensions = r.dimensions;
      this->n = r.n;
      this->allocate();
    }

    if(this->device != r.device || this->device != TensorLocation::GPU){
      throw std::runtime_error("Half to float conversion must have both tensors in gpu - T != U copy helper");
    }
    
    if constexpr(std::is_floating_point_v<T>){
      size_t s = this->size;
      const int thCount = 256, blocks = (s + thCount - 1) / thCount; //floors
      dim3 blockDim(thCount), gridDim(blocks);
      T* dst = this->deviceData.get();
      U* src = r.deviceData.get();
      HalfToFullKernel<<<gridDim, blockDim, 0, *r.stream>>>(dst, src, s);
    }
    else{
      size_t s = this->size;
      const int thCount = 256, blocks = (s + thCount - 1) / thCount; //floors
      dim3 blockDim(thCount), gridDim(blocks);
      T* dst = this->deviceData.get();
      U* src = r.deviceData.get();
      FullToHalfKernel<<<gridDim, blockDim, 0, *r.stream>>>(dst, src, s);
    }
  }

};

//deduction guides for copy constructor
template <typename T, AllocationType ALLOC>
Tensor(const Tensor<T, ALLOC>&) -> Tensor<T, ALLOC>;

//deduction guide for move constructor
template <typename T, AllocationType ALLOC>
Tensor(Tensor<T, ALLOC>&&) -> Tensor<T, ALLOC>;

#endif