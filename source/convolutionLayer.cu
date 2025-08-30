#include "../header/convolutionLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>

//think its good - this is not true at all

thread_local cudnnHandle_t Layer::nnHandle{};
thread_local cublasHandle_t Layer::blasHandle{};

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

__inline__ void tensorDesc(cudnnTensorDescriptor_t& desc, const Tensor& T){
  TryCuda(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], T.dimensions[2], T.dimensions[3]));
}

/*
bias and filter members
cudnnTensorDescriptor_t inputD, outputD; cudnnConvolutionDescriptor_t convoD;
cudnnFilterDescriptor_t filterD; cudnnTensorDescriptor_t biasD;
*/


ConvolutionLayer::ConvolutionLayer(const int fC, const int iC, const int fH, const int fW) : bias({fC}, TensorLocation::GPU), filters({fC, iC, fH, fW}, TensorLocation::GPU){
  TryCuda(cudnnCreateTensorDescriptor(&inputD));
  TryCuda(cudnnCreateTensorDescriptor(&outputD));
  TryCuda(cudnnCreateTensorDescriptor(&biasD));
  TryCuda(cudnnCreateFilterDescriptor(&filterD));
  TryCuda(cudnnCreateConvolutionDescriptor(&convoD));
  TryCuda(cudnnSetFilter4dDescriptor(filterD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fC, iC, fH, fW));
  TryCuda(cudnnSetTensor4dDescriptor(biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, fC, 1, 1));
}

Tensor ConvolutionLayer::forward(const Tensor& T, bool train){

  int n = T.dimensions[0], c = T.dimensions[1], h = T.dimensions[2], w = T.dimensions[3];
  input = Tensor(T);

  //setting descriptors, calculating output dimensions
  TryCuda(cudnnSetTensor4dDescriptor(inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //input
  TryCuda(cudnnGetConvolution2dForwardOutputDim(convoD, inputD, filterD, &n, &c, &h, &w)); //calculates the dimension sizes that the output will have
  TryCuda(cudnnSetTensor4dDescriptor(outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //output
  Tensor output({n, c, h, w}, TensorLocation::GPU); //readies return tensor
  //variables for convolution algorithm and workspace memory
  cudnnConvolutionFwdAlgoPerf_t potential;
  cudnnConvolutionFwdAlgo_t algo;
  int algoCount = 0;
  size_t wsSize = 0;
  void* workspace = nullptr;
  float a = 1.0f, b = 1.0f, b2 = 0.0f; //alpha betas
  //finds the best convolution algorithm to use
  TryCuda(cudnnFindConvolutionForwardAlgorithm(nnHandle, inputD, filterD, convoD, 
                                              outputD, 1, &algoCount, &potential));
  if(algoCount == 0){ //safety check if none are found
    throw std::runtime_error("cuDNN failed to find convolution");
  }
  algo = potential.algo; 
  //determines the necessary workspace size for convolution
  TryCuda(cudnnGetConvolutionForwardWorkspaceSize(nnHandle, inputD, filterD, 
                                                  convoD, outputD, algo, &wsSize));
  if(wsSize > 0){ //allocates necessary workspace space if any
    TryCuda(cudaMalloc((void**)&workspace, wsSize));
  }
  //performs convolution
  TryCuda(cudnnConvolutionForward(nnHandle, &mx, inputD, T.gpuData(), filterD, 
                          filters.gpuData(), convoD, algo, workspace, 
                          wsSize, &mn, outputD, output.gpuData()));
  //freeing memory
  if(workspace != nullptr){ //frees the workspace if it was used
    TryCuda(cudaFree(workspace));
  }

  //performs bias addition
  TryCuda(cudnnAddTensor(nnHandle, &a, biasD, bias.gpuData(), &b2, outputD, output.gpuData()));  
  return output;
}

Tensor ConvolutionLayer::backward(const Tensor& gradient){
  //initializing gradient tensors and descriptor parameters
  
  iGrad = Tensor(input.dimensions, TensorLocation::GPU, input.n);
  bGrad = Tensor(bias.dimensions, TensorLocation::GPU, bias.n);
  fGrad = Tensor(filters.dimensions, TensorLocation::GPU, filters.n);
  //bad naming, first two for filter, last two for the returned gradient
  cudnnConvolutionBwdFilterAlgoPerf_t potential;
  cudnnConvolutionBwdFilterAlgo_t algo;
  cudnnConvolutionBwdDataAlgoPerf_t dataPot;
  cudnnConvolutionBwdDataAlgo_t dataAlgo;

  int algoCount = 0;
  size_t wsSize = 0, wsSizeTmp;
  void* workspace = nullptr;
  //applies back propagation through the convolutions bias
  TryCuda(cudnnConvolutionBackwardBias(nnHandle, &mx, outputD, gradient.gpuData(), &mn, biasD, bGrad.gpuData()));
  TryCuda(cudnnFindConvolutionBackwardDataAlgorithm(nnHandle, filterD, outputD, convoD, inputD, 1, &algoCount, &dataPot));
  TryCuda(cudnnFindConvolutionBackwardFilterAlgorithm(nnHandle, inputD, outputD, convoD, filterD, 1, &algoCount, &potential));
  dataAlgo = dataPot.algo;
  algo = potential.algo;
  TryCuda(cudnnGetConvolutionBackwardDataWorkspaceSize(nnHandle, filterD, outputD, convoD, inputD, dataAlgo, &wsSize));
  TryCuda(cudnnGetConvolutionBackwardFilterWorkspaceSize(nnHandle, inputD, outputD, convoD, filterD, algo, &wsSizeTmp));
  wsSize = std::max(wsSize, wsSizeTmp);
  if(wsSize > 0){ //allocates space for workspace if necessary, uses the higher space requirement so both can use the same allocation on their turn
    TryCuda(cudaMalloc((void**)&workspace, wsSize));
  }

  TryCuda(cudnnConvolutionBackwardData(nnHandle, &mx, filterD, filters.gpuData(), outputD, 
                                      gradient.gpuData(), convoD, dataAlgo, workspace, wsSize, &mn, 
                                      inputD, iGrad.gpuData()));
  
  TryCuda(cudnnConvolutionBackwardFilter(nnHandle, &mx, inputD, input.gpuData(), outputD, gradient.gpuData(), 
                                        convoD, algo, workspace, wsSize, &mn, filterD, fGrad.gpuData()));
  if(workspace != nullptr){
    TryCuda(cudaFree(workspace));
    workspace = nullptr;
  }
  return iGrad;
}

ConvolutionLayer::~ConvolutionLayer(){
  TryCuda(cudnnDestroyFilterDescriptor(filterD));
  TryCuda(cudnnDestroyTensorDescriptor(inputD));
  TryCuda(cudnnDestroyTensorDescriptor(outputD));
  TryCuda(cudnnDestroyTensorDescriptor(biasD));
  TryCuda(cudnnDestroyConvolutionDescriptor(convoD));
}
