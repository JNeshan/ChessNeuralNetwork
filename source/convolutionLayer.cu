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
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t inputD, outputD, biasD;
  cudnnFilterDescriptor_t filterD;
  cudnnConvolutionDescriptor_t convoD;

  CudaMembers(){
    TryCuda(cudnnCreate(&handle));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
    TryCuda(cudnnCreateTensorDescriptor(&biasD));
    TryCuda(cudnnCreateConvolutionDescriptor(&convoD));
  }
  ~CudaMembers(){
  };
};

ConvolutionLayer::ConvolutionLayer(const int fC, const int iC, const int fH, const int fW) : bias({fC}, TensorLocation::GPU), filters({fC, iC, fW, fH}, TensorLocation::GPU) {
  CudaM = new CudaMembers();
  TryCuda(cudnnSetFilter4dDescriptor(CudaM->filterD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fC, iC, fH, fW));
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, fC, 1, 1));
  TryCuda(cudnnSetConvolution2dDescriptor(CudaM->convoD, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
}

Tensor ConvolutionLayer::forward(const Tensor& T){
  int n = T.dimensions[0], c = (T.dimensions[1] / ((int)pow(8, 4 - T.dimensions.size()))), 
  h = 8, w = 8;

  input = Tensor(T);
  if(input.dimensions.size() != 4){
    throw("Convolution bad input");
  }

  //setting descriptors, calculating output dimensions
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  TryCuda(cudnnGetConvolution2dForwardOutputDim(CudaM->convoD, CudaM->inputD, CudaM->filterD, &n, &c, &h, &w));
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  Tensor output({n, c, h, w}, TensorLocation::GPU);
  //variables for convolution algorithm and workspace memory
  cudnnConvolutionFwdAlgoPerf_t potential;
  cudnnConvolutionFwdAlgo_t algo;
  int algoCount = 0;
  size_t wsSize = 0;
  void* workspace = nullptr;
  float a = 1.0f, b = 1.0f, b2 = 0.0f; //alpha betas
  
  TryCuda(cudnnFindConvolutionForwardAlgorithm(CudaM->handle, CudaM->inputD, CudaM->filterD, CudaM->convoD, 
                                              CudaM->outputD, 1, &algoCount, &potential));
  if(algoCount == 0){
    throw std::runtime_error("cuDNN failed to find convolution");
  }
  algo = potential.algo;

  TryCuda(cudnnGetConvolutionForwardWorkspaceSize(CudaM->handle, CudaM->inputD, CudaM->filterD, 
                                                  CudaM->convoD, CudaM->outputD, algo, &wsSize));
  if(wsSize > 0){
    TryCuda(cudaMalloc((void**)&workspace, wsSize));
  }
  //performs convolution
  TryCuda(cudnnConvolutionForward(CudaM->handle, &a, CudaM->inputD, T.gpuData(), CudaM->filterD, 
                          filters.gpuData(), CudaM->convoD, algo, workspace, 
                          wsSize, &b, CudaM->outputD, output.gpuData()));
  //freeing memory
  if(workspace != nullptr){
    TryCuda(cudaFree(workspace));
  }

  //performs bias addition
  TryCuda(cudnnAddTensor(CudaM->handle, &a, CudaM->biasD, bias.gpuData(), &b2, CudaM->outputD, output.gpuData()));
  return output;
}

ConvolutionLayer::~ConvolutionLayer(){
  delete CudaM;
}
