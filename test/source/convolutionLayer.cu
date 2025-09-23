#include "../header/convolutionLayer.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>
#include <chrono>
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

ConvolutionLayer::ConvolutionLayer(const int fC, const int iC, const int fH, const int fW, const int pad) : forw(false), back(false), bias({1, fC}, TensorLocation::GPU), filters({fC, iC, fH, fW}, TensorLocation::GPU), fGrad({fC, iC, fH, fW}, TensorLocation::GPU), bGrad({1, fC}, TensorLocation::GPU), padding(pad){
  TryCuda(cudnnCreateTensorDescriptor(&inputD));
  TryCuda(cudnnCreateTensorDescriptor(&outputD));
  TryCuda(cudnnCreateTensorDescriptor(&biasD));
  TryCuda(cudnnCreateFilterDescriptor(&filterD));
  TryCuda(cudnnCreateConvolutionDescriptor(&convoD));
  TryCuda(cudnnCreateActivationDescriptor(&this->actD));
  TryCuda(cudnnSetFilter4dDescriptor(filterD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fC, iC, fH, fW));
  TryCuda(cudnnSetTensor4dDescriptor(biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, fC, 1, 1));
  TryCuda(cudnnSetConvolution2dDescriptor(convoD, padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  TryCuda(cudnnSetConvolutionMathType(this->convoD, CUDNN_TENSOR_OP_MATH));
  TryCuda(cudnnSetActivationDescriptor(this->actD, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  this->convoAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  this->wsPtrF = nullptr;
  this->wsPtrB = nullptr;
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& lay) : padding(lay.padding), filters(lay.filters), bias(lay.bias), fGrad(lay.fGrad), bGrad(lay.bGrad){
  const auto [k, c, h, w] = std::tie(lay.filters.dimensions[0], lay.filters.dimensions[1], lay.filters.dimensions[2], lay.filters.dimensions[3]);
  TryCuda(cudnnCreateTensorDescriptor(&this->inputD));
  TryCuda(cudnnCreateTensorDescriptor(&this->outputD));
  TryCuda(cudnnCreateTensorDescriptor(&this->biasD));
  TryCuda(cudnnCreateFilterDescriptor(&this->filterD));
  TryCuda(cudnnCreateConvolutionDescriptor(&this->convoD));
  TryCuda(cudnnCreateActivationDescriptor(&this->actD));
  TryCuda(cudnnSetFilter4dDescriptor(this->filterD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));
  TryCuda(cudnnSetTensor4dDescriptor(this->biasD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, k, 1, 1));
  TryCuda(cudnnSetConvolution2dDescriptor(this->convoD, this->padding, this->padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  TryCuda(cudnnSetConvolutionMathType(this->convoD, CUDNN_TENSOR_OP_MATH));
  TryCuda(cudnnSetActivationDescriptor(this->actD, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  this->convoAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  this->forw = false;
  this->back = false;
  this->wsSizeF = 0;
  this->wsSizeB = 0;
  this->wsPtrF = nullptr;
  this->wsPtrB = nullptr;
}

ConvolutionLayer::~ConvolutionLayer(){
  TryCuda(cudnnDestroyFilterDescriptor(this->filterD));
  TryCuda(cudnnDestroyTensorDescriptor(this->inputD));
  TryCuda(cudnnDestroyTensorDescriptor(this->outputD));
  TryCuda(cudnnDestroyTensorDescriptor(this->biasD));
  TryCuda(cudnnDestroyConvolutionDescriptor(this->convoD));
  if(this->wsPtrF != nullptr){ //frees the workspace if it was used
    TryCuda(cudaFree(this->wsPtrF));
  }
  if(this->wsPtrB != nullptr){
    TryCuda(cudaFree(this->wsPtrB));
  }
}


std::unique_ptr<Layer> ConvolutionLayer::clone(){
  return(std::make_unique<ConvolutionLayer>(*this));
}

Tensor ConvolutionLayer::forward(Tensor& T, bool train){
  auto start = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::steady_clock::now() - start;
  int n = T.dimensions[0], c = T.dimensions[1], h = T.dimensions[2], w = T.dimensions[3];
  if(filters.dimensions[1] != T.dimensions[1]){
    throw std::runtime_error("Incorrect input channels for convolutional layer");
  }
  TryCuda(cudnnSetTensor4dDescriptor(inputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //input
  TryCuda(cudnnGetConvolution2dForwardOutputDim(convoD, inputD, filterD, &n, &c, &h, &w)); //calculates the dimension sizes that the output will have
  TryCuda(cudnnSetTensor4dDescriptor(outputD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //output

  
  Tensor output({n, c, h, w}, TensorLocation::GPU); //readies return tensor
  
  //variables for convolution algorithm and workspace memory
  //void* workspace = nullptr;
  
  if(!this->forw){ //very time consuming to get the optimal algorithm and details so it only does it once, could replace with a map since it'll always be relative to batch size
    TryCuda(cudnnGetConvolutionForwardWorkspaceSize(this->nnHandle, this->inputD, this->filterD, 
                                                  this->convoD, this->outputD, this->convoAlgo, &this->wsSizeF));
    
    if(this->wsSizeF > 0){ //allocates necessary workspace space if any
      TryCuda(cudaMalloc((void**)&this->wsPtrF, this->wsSizeF));
    }

    this->forw = true;
  }

  //performs convolution
  //cudaDeviceSynchronize();
  //auto startL = std::chrono::steady_clock::now();
  TryCuda(cudnnConvolutionBiasActivationForward(this->nnHandle, &mx, this->inputD, T.gpuData(), this->filterD, this->filters.gpuData(), this->convoD, this->convoAlgo,
                                                this->wsPtrF, this->wsSizeF, &mn, this->outputD, output.gpuData(), this->biasD, this->bias.gpuData(), this->actD,
                                                this->outputD, output.gpuData()));
  //cudaDeviceSynchronize();
  //elapsed = std::chrono::steady_clock::now() - startL;
  ////std::cout<<std::string("Time in convolution op: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  

  if(train){
    this->input = std::move(T);
  }
  elapsed = std::chrono::steady_clock::now() - start;
  //std::cout<<std::string("Time in convolution: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  return std::move(output);
}

Tensor ConvolutionLayer::backward(Tensor& gradient){
  auto start = std::chrono::steady_clock::now();
  //Tensor iGrad = Tensor(this->input.dimensions, TensorLocation::GPU);
  
  //initializing gradient tensors and descriptor parameters
  
  //bad naming, first two for filter, last two for the returned gradient
  
  //applies back propagation through the convolutions bias

  if(!this->back){
    int algoCount = 0;
    size_t wsSizeX = 0, wsSizeY;
    cudnnConvolutionBwdFilterAlgoPerf_t potential;
    cudnnConvolutionBwdDataAlgoPerf_t dataPot;
    TryCuda(cudnnFindConvolutionBackwardDataAlgorithm(this->nnHandle, this->filterD, this->outputD, this->convoD, this->inputD, 1, &algoCount, &dataPot));
    TryCuda(cudnnFindConvolutionBackwardFilterAlgorithm(this->nnHandle, this->inputD, this->outputD, this->convoD, this->filterD, 1, &algoCount, &potential));
    this->backDataAlgo = dataPot.algo;
    this->backFilterAlgo = potential.algo;
    TryCuda(cudnnGetConvolutionBackwardDataWorkspaceSize(this->nnHandle, this->filterD, this->outputD, this->convoD, this->inputD, this->backDataAlgo, &wsSizeX));
    TryCuda(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->nnHandle, this->inputD, this->outputD, this->convoD, this->filterD, this->backFilterAlgo, &wsSizeY));
    
    
    this->wsSizeB = std::max(wsSizeX, wsSizeY);
    if(this->wsSizeB > 0){ //allocates space for workspace if necessary, uses the higher space requirement so both can use the same allocation on their turn
      TryCuda(cudaMalloc((void**)&this->wsPtrB, this->wsSizeB));
    }
    this->back = true;
  }
  TryCuda(cudnnConvolutionBackwardBias(this->nnHandle, &mx, this->outputD, gradient.gpuData(), &mn, this->biasD, this->bGrad.gpuData()));

  TryCuda(cudnnConvolutionBackwardFilter(this->nnHandle, &mx, this->inputD, this->input.gpuData(), this->outputD, gradient.gpuData(), 
                                        this->convoD, this->backFilterAlgo, this->wsPtrB, this->wsSizeB, &mn, this->filterD, fGrad.gpuData()));

  TryCuda(cudnnConvolutionBackwardData(this->nnHandle, &mx, this->filterD, this->filters.gpuData(), this->outputD, 
                                      gradient.gpuData(), this->convoD, this->backDataAlgo, this->wsPtrB, this->wsSizeB, &mn, 
                                      this->inputD, this->input.gpuData()));

  //TryCuda(cudnnConvolutionBackwardData(nnHandle, &mx, filterD, filters.gpuData(), outputD, 
  //                                    gradient.gpuData(), convoD, dataAlgo, workspace, wsSize, &mn, 
  //                                    inputD, iGrad.gpuData()));
  auto elapsed = std::chrono::steady_clock::now() - start;
  //std::cout<<std::string("Time in convolution back: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  
  return std::move(this->input);
}

void ConvolutionLayer::genTensorData(){
  Generator::tGen(bias);
  Generator::tGen(filters);
}

void ConvolutionLayer::loadTensor(std::ifstream& iF){
  if(!iF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }

  filters.readBinary(iF);
  bias.readBinary(iF);
  filters.gpuSend();
  bias.gpuSend();
}

void ConvolutionLayer::saveTensor(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }

  filters.writeBinary(oF);
  bias.writeBinary(oF);
  filters.gpuSend();
  bias.gpuSend();
}

void ConvolutionLayer::cleanSave(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }
  oF << "Convolutional Layer Tensors:\nFilters ";
  filters.writeTensor(oF);
  oF<<"Bias ";
  bias.writeTensor(oF);
  filters.gpuSend();
  bias.gpuSend();
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> ConvolutionLayer::getLearningData(){
  return {{&filters, &bias}, {&fGrad, &bGrad}};
}

