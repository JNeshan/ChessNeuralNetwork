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

/*
bias and filter members
cudnnTensorDescriptor_t inputD, outputD; cudnnConvolutionDescriptor_t convoD;
cudnnFilterDescriptor_t filterD; cudnnTensorDescriptor_t biasD;
*/

ConvolutionLayer::ConvolutionLayer(const int fC, const int fH, const int iC, const int fW, const int pad) : forw(false), back(false), bias({1, 1, 1, fC}, TensorLocation::GPU, 1),
                                  filters({fC, fH, fW, iC}, TensorLocation::GPU), fGrad({fC, fH, fW, iC}, TensorLocation::GPU), bGrad({1, 1, 1, fC}, TensorLocation::GPU, 1), padding(pad){
  TryCuda(cudnnCreateTensorDescriptor(&this->inputD));
  TryCuda(cudnnCreateTensorDescriptor(&this->outputD));
  TryCuda(cudnnCreateTensorDescriptor(&this->biasD));
  TryCuda(cudnnCreateFilterDescriptor(&this->filterD));
  TryCuda(cudnnCreateConvolutionDescriptor(&this->convoD));
  TryCuda(cudnnCreateActivationDescriptor(&this->actD));
  TryCuda(cudnnSetConvolution2dDescriptor(this->convoD, this->padding, this->padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  TryCuda(cudnnSetConvolutionMathType(this->convoD, CUDNN_TENSOR_OP_MATH));
  TryCuda(cudnnSetActivationDescriptor(this->actD, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  this->convoAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  this->filters.genDescriptor(&this->filterD);
  this->bias.genDescriptor(&this->biasD);
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& lay) : padding(lay.padding), filters(lay.filters), bias(lay.bias), fGrad(lay.fGrad), bGrad(lay.bGrad){
  const auto [k, h, w, c] = std::tie(lay.filters.dimensions[0], lay.filters.dimensions[1], lay.filters.dimensions[2], lay.filters.dimensions[3]);
  TryCuda(cudnnCreateTensorDescriptor(&this->inputD));
  TryCuda(cudnnCreateTensorDescriptor(&this->outputD));
  TryCuda(cudnnCreateTensorDescriptor(&this->biasD));
  TryCuda(cudnnCreateFilterDescriptor(&this->filterD));
  TryCuda(cudnnCreateConvolutionDescriptor(&this->convoD));
  TryCuda(cudnnCreateActivationDescriptor(&this->actD));
  this->filters.genDescriptor(&this->filterD);
  this->bias.genDescriptor(&this->biasD);
  TryCuda(cudnnSetConvolution2dDescriptor(this->convoD, this->padding, this->padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  TryCuda(cudnnSetConvolutionMathType(this->convoD, CUDNN_TENSOR_OP_MATH));
  TryCuda(cudnnSetActivationDescriptor(this->actD, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  this->convoAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  this->forw = false;
  this->back = false;
}

ConvolutionLayer::~ConvolutionLayer(){
  TryCuda(cudnnDestroyFilterDescriptor(this->filterD));
  TryCuda(cudnnDestroyTensorDescriptor(this->inputD));
  TryCuda(cudnnDestroyTensorDescriptor(this->outputD));
  TryCuda(cudnnDestroyTensorDescriptor(this->biasD));
  TryCuda(cudnnDestroyConvolutionDescriptor(this->convoD));
}


std::unique_ptr<Layer> ConvolutionLayer::clone(){
  return(std::make_unique<ConvolutionLayer>(*this));
}

Tensor<__half> ConvolutionLayer::forward(Tensor<__half>& T, bool train){
  auto start = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::steady_clock::now() - start;
  //auto startL = std::chrono::steady_clock::now();
  //cudaDeviceSynchronize();
  //elapsed = std::chrono::steady_clock::now() - startL;
  ////std::cout<<std::string("Time in convolution op: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  

  if(filters.dimensions[3] != T.dimensions[3]){
    throw std::runtime_error("Convolution forward recieved invalid tensor input channels");
  }

  if(!this->forw){ //first run only
    this->input = T;
    this->input.genDescriptor(&this->inputD);
    
    TryCuda(cudnnGetConvolution2dForwardOutputDim(convoD, inputD, filterD, &this->n, &this->c, &this->h, &this->w));
    TryCuda(cudnnSetTensor4dDescriptor(this->outputD, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w));
    size_t tmpSize;
    TryCuda(cudnnGetConvolutionForwardWorkspaceSize(this->nnHandle, this->inputD, this->filterD, 
                                                  this->convoD, this->outputD, this->convoAlgo, &tmpSize));
    
    if(tmpSize > Layer::wsSize){
      if(Layer::wsPtr){
        TryCuda(cudaFree(Layer::wsPtr));
      }
      Layer::wsSize = tmpSize;
      TryCuda(cudaMalloc((void**)&Layer::wsPtr, Layer::wsSize));
    }
  }

  Tensor<__half> output({this->n, this->h, this->w, this->c}, TensorLocation::GPU, 4); //prepares output tensor in memory

  if(!this->forw){
    this->forw = true;
    output.genDescriptor(&this->outputD);
  }

  TryCuda(cudnnConvolutionBiasActivationForward(this->nnHandle, &mx, this->inputD, T.gpuData(), this->filterD, this->filters.gpuData(), this->convoD, this->convoAlgo,
                                                Layer::wsPtr, Layer::wsSize, &mn, this->outputD, output.gpuData(), this->biasD, this->bias.gpuData(), this->actD,
                                                this->outputD, output.gpuData()));
  return output;
}

BackwardPackage ConvolutionLayer::backward(Tensor<__half>& gradient){
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
    
    
    wsSizeX = std::max(wsSizeX, wsSizeY);
    if(wsSizeX > Layer::wsSize){
      if(Layer::wsPtr){
        TryCuda(cudaFree(Layer::wsPtr));
      }
      Layer::wsSize = wsSizeX;
      TryCuda(cudaMalloc((void**)&Layer::wsPtr, Layer::wsSize));
    }
    this->back = true;
  }
  TryCuda(cudnnConvolutionBackwardBias(this->nnHandle, &mx, this->outputD, gradient.gpuData(), &mn, this->biasD, this->bGrad.gpuData()));

  TryCuda(cudnnConvolutionBackwardFilter(this->nnHandle, &mx, this->inputD, this->input.gpuData(), this->outputD, gradient.gpuData(), 
                                        this->convoD, this->backFilterAlgo, Layer::wsPtr, Layer::wsSize, &mn, this->filterD, fGrad.gpuData()));

  TryCuda(cudnnConvolutionBackwardData(this->nnHandle, &mx, this->filterD, this->filters.gpuData(), this->outputD, 
                                      gradient.gpuData(), this->convoD, this->backDataAlgo, Layer::wsPtr, Layer::wsSize, &mn, 
                                      this->inputD, this->input.gpuData()));

  //TryCuda(cudnnConvolutionBackwardData(nnHandle, &mx, filterD, filters.gpuData(), outputD, 
  //                                    gradient.gpuData(), convoD, dataAlgo, workspace, wsSize, &mn, 
  //                                    inputD, iGrad.gpuData()));
  auto elapsed = std::chrono::steady_clock::now() - start;
  //std::cout<<std::string("Time in convolution back: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  
  return std::move(this->input);
}

void ConvolutionLayer::genTensorData(){
  Tensor<float> tmpB(bias), tmpF(filters);
  Generator::tGen(tmpB);
  Generator::tGen(tmpF);

  bias = tmpB;
  filters = tmpF;
}

void ConvolutionLayer::loadTensor(std::ifstream& iF){
  if(!iF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }

  filters.readBinary(iF);
  bias.readBinary(iF);
}

void ConvolutionLayer::saveTensor(std::ofstream& oF){
  if(!oF.is_open()){
    std::cout<<"File not open"<<std::endl;
    return;
  }

  filters.writeBinary(oF);
  bias.writeBinary(oF);
}

//void ConvolutionLayer::cleanSave(std::ofstream& oF){
//  if(!oF.is_open()){
//    std::cout<<"File not open"<<std::endl;
//    return;
//  }
//  oF << "Convolutional Layer Tensors:\nFilters ";
//  filters.writeTensor(oF);
//  oF<<"Bias ";
//  bias.writeTensor(oF);
//  filters.gpuSend();
//  bias.gpuSend();
//}


