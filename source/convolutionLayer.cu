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
    TryCuda(cudnnCreateTensorDescriptor(&gradientD));

  }
  ~CudaMembers(){
    TryCuda(cudnnDestroyTensorDescriptor(inputD));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnDestroyTensorDescriptor(biasD));
    TryCuda(cudnnDestroyConvolutionDescriptor(convoD));
    TryCuda(cudnnDestroy(handle));
    TryCuda(cudnnDestroyTensorDescriptor(gradientD));
  };

  void resetTemp(){
    TryCuda(cudnnDestroyTensorDescriptor(inputD));
    TryCuda(cudnnDestroyTensorDescriptor(outputD));
    TryCuda(cudnnDestroyConvolutionDescriptor(convoD));
    TryCuda(cudnnCreateTensorDescriptor(&inputD));
    TryCuda(cudnnCreateTensorDescriptor(&outputD));
    TryCuda(cudnnCreateConvolutionDescriptor(&convoD));
  }
};

ConvolutionLayer::ConvolutionLayer(const int fC, const int iC, const int fH, const int fW) : bias({fC}, TensorLocation::GPU), filters({fC, iC, fW, fH}, TensorLocation::GPU){
  CudaM = new CudaMembers();
  TryCuda(cudnnSetFilter4dDescriptor(CudaM->filterD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fC, iC, fH, fW));
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, fC, 1, 1));
  TryCuda(cudnnSetConvolution2dDescriptor(CudaM->convoD, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
}

Tensor ConvolutionLayer::forward(const Tensor& T){
  int n = T.dimensions[0], c = T.dimensions[1], h = T.dimensions[2], w = T.dimensions[3]; //dimension variables, reused for memlocations in cudnn calls
  input = Tensor(T); //copies input tensor for backpropagation

  //setting descriptors, calculating output dimensions
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  TryCuda(cudnnGetConvolution2dForwardOutputDim(CudaM->convoD, CudaM->inputD, CudaM->filterD, &n, &c, &h, &w));
  TryCuda(cudnnSetTensor4dDescriptor(CudaM->outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  Tensor output({n, c, h, w}, TensorLocation::GPU); //readies return tensor
  //variables for convolution algorithm and workspace memory
  cudnnConvolutionFwdAlgoPerf_t potential;
  cudnnConvolutionFwdAlgo_t algo;
  int algoCount = 0;
  size_t wsSize = 0;
  void* workspace = nullptr;
  float a = 1.0f, b = 1.0f, b2 = 0.0f; //alpha betas
  //finds the best convolution algorithm to use
  TryCuda(cudnnFindConvolutionForwardAlgorithm(CudaM->handle, CudaM->inputD, CudaM->filterD, CudaM->convoD, 
                                              CudaM->outputD, 1, &algoCount, &potential));
  if(algoCount == 0){ //safety check if none are found
    throw std::runtime_error("cuDNN failed to find convolution");
  }
  algo = potential.algo; 
  //determines the necessary workspace size for convolution
  TryCuda(cudnnGetConvolutionForwardWorkspaceSize(CudaM->handle, CudaM->inputD, CudaM->filterD, 
                                                  CudaM->convoD, CudaM->outputD, algo, &wsSize));
  if(wsSize > 0){ //allocates necessary workspace space if any
    TryCuda(cudaMalloc((void**)&workspace, wsSize));
  }
  //performs convolution
  TryCuda(cudnnConvolutionForward(CudaM->handle, &a, CudaM->inputD, T.gpuData(), CudaM->filterD, 
                          filters.gpuData(), CudaM->convoD, algo, workspace, 
                          wsSize, &b, CudaM->outputD, output.gpuData()));
  //freeing memory
  if(workspace != nullptr){ //frees the workspace if it was used
    TryCuda(cudaFree(workspace));
  }

  //performs bias addition
  TryCuda(cudnnAddTensor(CudaM->handle, &a, CudaM->biasD, bias.gpuData(), &b2, CudaM->outputD, output.gpuData()));
  return output;
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> ConvolutionLayer::backward(const Tensor& gradient){
  //initializing gradient tensors and descriptor parameters
  iGrad = Tensor(input.dimensions, TensorLocation::GPU);
  bGrad = Tensor(bias.dimensions, TensorLocation::GPU);
  fGrad = Tensor(filters.dimensions, TensorLocation::GPU);

  cudnnConvolutionBwdFilterAlgoPerf_t potential;
  cudnnConvolutionBwdFilterAlgo_t algo;
  cudnnConvolutionBwdDataAlgoPerf_t dataPot;
  cudnnConvolutionBwdDataAlgo_t dataAlgo;
  int algoCount = 0;
  size_t wsSize = 0, wsSizeTmp;
  void* workspace = nullptr;
  //applies back propagation through the convolutions bias
  TryCuda(cudnnConvolutionBackwardBias(CudaM->handle, &mx, CudaM->outputD, gradient.gpuData(), &mn, CudaM->outputD, bGrad.gpuData()));
  TryCuda(cudnnFindConvolutionBackwardDataAlgorithm(CudaM->handle, CudaM->filterD, CudaM->outputD, CudaM->convoD, CudaM->inputD, 1, &algoCount, &dataPot));
  TryCuda(cudnnFindConvolutionBackwardFilterAlgorithm(CudaM->handle, CudaM->inputD, CudaM->outputD, CudaM->convoD, CudaM->filterD, 1, &algoCount, &potential));
  dataAlgo = dataPot.algo;
  algo = potential.algo;
  TryCuda(cudnnGetConvolutionBackwardDataWorkspaceSize(CudaM->handle, CudaM->filterD, CudaM->outputD, CudaM->convoD, CudaM->inputD, dataAlgo, &wsSize));
  TryCuda(cudnnGetConvolutionBackwardFilterWorkspaceSize(CudaM->handle, CudaM->inputD, CudaM->outputD, CudaM->convoD, CudaM->filterD, algo, &wsSizeTmp));
  wsSize = std::max(wsSize, wsSizeTmp);
  if(wsSize > 0){
    TryCuda(cudaMalloc((void**)&workspace, wsSize));
  }

  TryCuda(cudnnConvolutionBackwardData(CudaM->handle, &mx, CudaM->filterD, filters.gpuData(), CudaM->outputD, 
                                      gradient.gpuData(), CudaM->convoD, dataAlgo, workspace, wsSize, &mn, 
                                      CudaM->inputD, iGrad.gpuData()));
  
  TryCuda(cudnnConvolutionBackwardFilter(CudaM->handle, &mx, CudaM->inputD, input.gpuData(), CudaM->outputD, gradient.gpuData(), 
                                        CudaM->convoD, algo, workspace, wsSize, &mn, CudaM->filterD, fGrad.gpuData()));
  if(workspace != nullptr){
    TryCuda(cudaFree(workspace));
    workspace = nullptr;
  }

  return {{&input, &filters, &bias}, {&iGrad, &fGrad, &bGrad}}; //fix output
}

ConvolutionLayer::~ConvolutionLayer(){
  delete CudaM;
}
