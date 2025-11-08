//tensor.cuh
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <memory>
#include <cuda_fp16.h>
#include <type_traits>
#include <string>
#include <sstream>
#include <iomanip>
#include "cudnn.h"
#include "cuda_runtime.h"
#include "cublas.h"
#include "curand.h"

enum class TensorLocation {CPU, GPU};
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
    float sum = A[idx] + B[idx]; //will implicitly convert any half to float
    out[idx] = __float2half_ru(sum); //converts back to half
  }
  else{
    out[idx] = A[idx] + B[idx]; //implicitly converts so it works
  }
}

//custom deleter for unique_ptr<T> in GPU memory (CUDA)
template <typename T>
struct CudaDeleter{
  void operator()(T* p) const noexcept{ //called whenever the object owned by unique_ptr<T, CudaDeleter> is deleted with the pointer to the T value contianed passed as p
    if(p) cudaFree(p); //frees if allocated
  }
};

template <typename T> 
class Tensor {
public:

    static_assert(std::is_floating_point_v<T> || std::is_same_v<T,__half>, "Tensor<T>: T must be a floating point type or half"); //enforces data type restriction

    //blank constructor
    Tensor() : dimensions(4), device(TensorLocation::GPU), n(-1), size(-1), hostData(nullptr), deviceData(nullptr){}

    Tensor(const std::vector<int>& dim, const TensorLocation loc = TensorLocation::CPU, int nth = 4) : dimensions(dim), device(loc), n(nth), hostData(nullptr), deviceData(nullptr){
      if(dim.size() != 4){
        throw std::runtime_error("Given dimensions does not have 4 elements");
      }
      this->size = 1;
      for(auto x : dim){ //computes size and ensures valid dimensions
        if(x <= 0){
          throw std::runtime_error("Non-positive dimension given");
        }
        this->size *= x;
      }
      if(loc == TensorLocation::CPU){
        if constexpr(!std::is_floating_point_v<T>){ //if stored on cpu must be full float
          throw std::runtime_error("Non-float tensor constructed to CPU");
        }
        this->hostData = std::make_unique<T[]>(this->size); //allocating space
      }
      else{
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
        this->deviceData.reset(tmpPtr);  
      }
    }

    //deep copy constructor
    template<typename U>
    Tensor(const Tensor<U>& r){
      this->size = r.size; //basic members
      this->device = r.device;
      this->dimensions = r.dimensions;
      this->n = r.n;

      if constexpr(std::is_same_v<T, U>){ //checks if both objects templated types are the same
        if(this->device == TensorLocation::CPU){
          this->hostData = std::make_unique<U[]>(this->size); //allocating memory then copying over on CPU
          memcpy(this->hostData.get(), r.hostData.get(), this->size * sizeof(T));
          this->deviceData = nullptr; //consistency
        }
        else{
          T* tmpPtr = nullptr; //allocating memory then copying over on the GPU
          TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T))); 
          this->deviceData.reset(tmpPtr);
          TryCudaTen(cudaMemcpy(this->deviceData.get(), r.deviceData.get(), this->size * sizeof(T), cudaMemcpyDeviceToDevice));
          this->hostData = nullptr;
        }
      }
      else{
        if(this->device != TensorLocation::GPU){ //half should only ever be on GPU
          throw std::runtime_error("Copied tensor using half stored on CPU");  
        }
        T* tmpPtr = nullptr; //allocating memory then calling appropriate conversion function
        TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
        this->deviceData.reset(tmpPtr);
        if constexpr(std::is_floating_point_v<U>){ //copied tensor is full, this is half
          convertFloatToHalf(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        else{ //converse
          convertHalfToFloat(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        this->hostData = nullptr;
      }
    }

    //move constructor, does not benefit over deep when converting data type
    template <typename U>
    Tensor(Tensor<U>&& r){
      this->size = r.size; //basic members
      this->device = r.device;
      this->dimensions = r.dimensions;
      this->n = r.n;
      if constexpr(std::is_same_v<T, U>){ //checking if both tensors use the same templated data type, performs fast ptr swap if so
        this->hostData.swap(r.hostData);
        this->deviceData.swap(r.deviceData);
      }
      else{ //if the templated data types differ, data type conversion is required
        //so a move operation between data types is as slow as a regular deep copy
        if(this->device != TensorLocation::GPU){ //half should only ever be on gpu
          throw std::runtime_error("Moved tensor using half stored on CPU");  
        }
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
        this->deviceData.reset(tmpPtr);
        if constexpr(std::is_floating_point_v<U>){ //copied tensor is full, this is half
          convertFloatToHalf(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        else{ //converse
          convertHalfToFloat(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        this->hostData = nullptr;
      }
    }
    
    ~Tensor(){} //destructor

    //move assignment operator overload
    template <typename U>
    Tensor<T>& operator=(Tensor<U>&& r){
      this->size = r.size;
      this->dimensions = r.dimensions;
      this->n = r.n;
      this->device = r.device;

      if constexpr(std::is_same_v<T, U>){ //checking if both tensors use the same templated data type, performs fast ptr swap if so
        if(this == &r){
          return *this; //preventing copying itself
        }
        this->hostData.swap(r.hostData);
        this->deviceData.swap(r.deviceData);
      }
      //converting between data types will always require a deep copy
      else{//if the templated data types differ, data type conversion is required
        //so a move operation between data types is as slow as a regular deep copy
        if(this->device == TensorLocation::CPU){
          throw std::runtime_error("Half stored on CPU in assignment operator overload call");
        }
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
        this->deviceData.reset(tmpPtr); 
        if constexpr(std::is_floating_point_v<U>){ //copied tensor is full, this is half
          convertFloatToHalf(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        else{ //converse
          convertHalfToFloat(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        this->hostData = nullptr;
      }
      return *this;
    }

    template <typename U>
    Tensor<T>& operator=(const Tensor<U>& r){
      this->size = r.size;
      this->dimensions = r.dimensions;
      this->n = r.n;
      
      if constexpr(std::is_same_v<T, U>){
        if(this == &r){ //flag
          return *this;
        }
        if(this->device == TensorLocation::GPU){
          T* tmpPtr = nullptr;
          TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
          this->deviceData.reset(tmpPtr);
          TryCudaTen(cudaMemcpy(this->deviceData.get(), r.deviceData.get(), this->size * sizeof(T), cudaMemcpyDeviceToDevice));
          this->hostData = nullptr;
        }
        else{
          this->hostData = std::make_unique<T[]>(this->size);
          memcpy(this->hostData.get(), r.hostData.get(), this->size * sizeof(T));
          this->deviceData = nullptr;
        }
      }
      else{ //if different data types, allocates space then calls the correct coversion function
        if(this->device == TensorLocation::CPU){
          throw std::runtime_error("Half stored on CPU in assignment operator overload call");
        }
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
        this->deviceData.reset(tmpPtr);
        if constexpr(std::is_floating_point_v<U>){ //copied tensor is full, this is half
          convertFloatToHalf(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        else{ //converse
          convertHalfToFloat(this->deviceData.get(), r.deviceData.get(), this->size);
        }
        this->hostData = nullptr;
      }
      return *this;
    }
        

    void gpuAdd(Tensor<T>& B){
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
      AddKernel<<<gridDim, blockDim>>>(aData, bData, aData, this->size);
      TryCudaTen(cudaDeviceSynchronize()); //templated kernel call

    }

    //addition operator overload performs element-wise addition over the tensor matrices
    Tensor<T> operator+(const Tensor<T>& rT) const {
      if(this->size != rT.size){
        throw std::runtime_error("Addition operator overload recieved incompatible tensors");
      }
      if(this->device != rT.device){
        throw std::runtime_error("Addition operator overload recieved tensors on seperate memory");
      }
      if(this->size <= 0){
        throw std::runtime_error("Unpopulated tensor passed to operator overloard");
      }
      Tensor<T> out(this->dimensions, this->device, this->n);
      if(this->device == TensorLocation::CPU){
        if constexpr(!std::is_floating_point_v<T>){
          throw std::runtime_error("Half stored on CPU in addition overload");
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
        AddKernel<<<gridDim, blockDim>>>(aData, bData, outPtr, this->size);
        TryCudaTen(cudaDeviceSynchronize()); //templated kernel call
      }
      return out;
    }
    //overloads += operator to perform element wise addition, tensors must have equal size
    template <typename U>
    Tensor<T>& operator+=(const Tensor<U>& rT){
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
          AddKernel<<<gridDim, blockDim>>>(aData, bData, aData, this->size);
          TryCudaTen(cudaDeviceSynchronize()); //templated kernel call
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
          AddKernel<<<gridDim, blockDim>>>(aData, bData, aData, this->size);
          TryCudaTen(cudaDeviceSynchronize()); //templated kernel call
      }
      return *this;
    }

    template <typename U>
    std::enable_if_t<std::is_same_v<T, U>, void>
    batchBuild(const std::vector<Tensor<U>>& tensorBatch, const TensorLocation loc = TensorLocation::CPU){
      if(tensorBatch.empty()){
        throw std::runtime_error("Set of passed tensors to batch empty");
      }
      this->n = 4; //setting basic members
      this->dimensions = tensorBatch[0].dimensions;
      this->dimensions[0] = tensorBatch.size();
      this->device = loc;
      this->size = tensorBatch[0].size;
      const int gap = this->size; //the number of elements per tensor being batched
      this->size *= tensorBatch.size();
      int offset = 0; //"this" ptr offset value
      if(loc == TensorLocation::CPU){
        if constexpr(!std::is_floating_point_v<T>){
          throw std::runtime_error("Half stored on CPU in batchBuild");
        }
        this->hostData = std::make_unique<T[]>(this->size);
        T* refPtr = this->hostData.get();
        for(auto& tRef : tensorBatch){ //performs memcpy over every tensor in the set, uses int offset to move the starting location to the first unused element
          if(tRef.size != gap){
            throw std::runtime_error("Tensor batch contains differently sized tensors");
          }
          memcpy(refPtr + offset, tRef.hostData.get(), gap * sizeof(T));
          offset += gap; //incrememnts the offset by however many elements were added
        }
      }
      else{
        T* tmpPtr = nullptr;
        TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
        this->deviceData.reset(tmpPtr);
        tmpPtr = this->deviceData.get();
        for(auto& tRef : tensorBatch){ //performs cudaMemcpy over every tensor in the set, uses int offset to move the starting location to the first unused element
          if(tRef.size != gap){
            throw std::runtime_error("Tensor batch contains differently sized tensors");
          }
          TryCudaTen(cudaMemcpy(tmpPtr + offset, tRef.gpuData(), gap * sizeof(T), cudaMemcpyDeviceToDevice));
          offset += gap;//incrememnts the offset by however many elements were added
        }
      }
    }

    Tensor<T> segment(const int n_i){
      if(n_i < 0 || n_i >= this->dimensions[0]){
        throw std::runtime_error("Bad batch index to copy");
      }
      std::vector<int> segDim = this->dimensions; //each tensor in a batch has the same dimensions with the dimensions[0] always being a tensors batch size
      segDim[0] = 1;
      Tensor<T> outTensor(segDim, this->device, this->n);
      int offset = n_i * outTensor.size; //marks offset of desired tensor
      if(this->device == TensorLocation::CPU){
        if constexpr(!std::is_floating_point_v<T>){
          throw std::runtime_error("Half stored on CPU in segment");
        }
        memcpy(outTensor.hostData.get(), this->hostData.get() + offset, outTensor.size * sizeof(T));
      }
      else{
        TryCudaTen(cudaMemcpy(outTensor.gpuData(), this->deviceData.get() + offset, outTensor.size * sizeof(T), cudaMemcpyDeviceToDevice));
      }
      return outTensor;
    }
    //takes cuda half pointer set and converts the values to float and stores it in dst
    void convertHalfToFloat(float* dst, const __half* src, int size){
      const int thCount = 256, blocks = (this->size + thCount - 1) / thCount; //floors
      dim3 blockDim(thCount), gridDim(blocks);
      HalfToFullKernel<<<gridDim, blockDim>>>(dst, src, size);
      cudaDeviceSynchronize();
    }
    //takes float pointer set and converts the values to cuda half and stores it in dst
    void convertFloatToHalf(__half* dst, const float* src, int size){
      const int thCount = 256, blocks = (this->size + thCount - 1) / thCount; //floors
      dim3 blockDim(thCount), gridDim(blocks);
      FullToHalfKernel<<<gridDim, blockDim>>>(dst, src, size);
      cudaDeviceSynchronize();
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
        return this->hostData[indice];
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
        return this->hostData[i];
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
      if constexpr(std::is_floating_point_v<T>){
        TryCudaTen(cudnnSetTensor4dDescriptor(*dsc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, this->dimensions[0], this->dimensions[3], this->dimensions[1], this->dimensions[2]));
      }
      else{
        TryCudaTen(cudnnSetTensor4dDescriptor(*dsc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, this->dimensions[0], this->dimensions[3], this->dimensions[1], this->dimensions[2]));  
      }
    }

    template <typename U = T>
    std::enable_if_t<std::is_same_v<U, __half>, void>
    genDescriptor(cudnnFilterDescriptor_t* dsc){
      if(this->size <= 0){
        throw std::runtime_error("Empty tensor generating descriptor");
      }
      TryCudaTen(cudnnSetFilter4dDescriptor(*dsc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, this->dimensions[0], this->dimensions[3], this->dimensions[1], this->dimensions[2]));
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
      this->hostData = std::make_unique<T[]>(this->size); //allocating host memory
      TryCudaTen(cudaMemcpy(this->hostData.get(), this->deviceData.get(), this->size * sizeof(T), cudaMemcpyDeviceToHost)); //copies to host
      this->deviceData.reset(); //frees the old data
      this->device = TensorLocation::CPU;
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
      T* tmpPtr = nullptr;
      TryCudaTen(cudaMalloc((void**)&tmpPtr, this->size * sizeof(T)));
      this->deviceData.reset(tmpPtr);
      TryCudaTen(cudaMemcpy(this->deviceData.get(), this->hostData.get(), this->size * sizeof(T), cudaMemcpyHostToDevice));
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
    std::enable_if_t<std::is_floating_point_v<U>>
    readBinary(std::ifstream& iF){
      if(!iF.is_open()){
        std::cout<<"File not open to read"<<std::endl;
        return;
      }
      this->deviceData = nullptr; //clearing existing data
      this->hostData = nullptr;
      this->dimensions = std::vector<int>(4); 
      this->device = TensorLocation::CPU;
      iF.read(reinterpret_cast<char*>(&this->size), sizeof(int));
      this->hostData = std::make_unique<T[]>(this->size); //allocate with new size
      iF.read(reinterpret_cast<char*>(&this->n), sizeof(int));
      iF.read(reinterpret_cast<char*>(this->dimensions.data()), 4 * sizeof(int));
      iF.read(reinterpret_cast<char*>(this->hostData.get()), this->size * sizeof(float));
    }
    //writes the tensors into binary assuming a binary output file, writes to wherever the output stream is so sequential tensor writes to the same open file can be read sequentially later
    template <typename U = T>
    std::enable_if_t<std::is_floating_point_v<U>> //only enabled if templated type is of float
    writeBinary(std::ofstream& oF){
      if(!oF.is_open()){
        std::cout<<"Output not open"<<std::endl;
        return;
      }
      if(this->size <= 0){
        std::cout<<"Tensor not allocated"<<std::endl;
        return;
      }
      if(this->device == TensorLocation::GPU){
        this->cpuSend();
      }
      oF.write(reinterpret_cast<const char*>(&this->size), sizeof(int));
      oF.write(reinterpret_cast<const char*>(&this->n), sizeof(int));
      oF.write(reinterpret_cast<const char*>(this->dimensions.data()), 4 * sizeof(int));
      oF.write(reinterpret_cast<const char*>(this->hostData.get()), this->size * sizeof(float));
    }
    //clean write for readability, converts all 2 or greater dimensional tensors to 2d for the display
    template <typename U = T>
    std::enable_if_t<std::is_floating_point_v<U>>
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

    //public member variables
    int size; //total number of elements and the highest order dimension
    int n; //nD tensor
    std::vector<int> dimensions; //dimension sizes
    //std::unique_ptr<T> data;
    std::unique_ptr<T[]> hostData;
    std::unique_ptr<T, CudaDeleter<T>> deviceData;

private:
    //enum to designate the CPU and GPU
    TensorLocation device;
};

#endif