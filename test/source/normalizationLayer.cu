#include "../header/normalizationLayer.h"
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

NormalizationLayer::NormalizationLayer(bool conv, int channels) : scale({1, channels}, TensorLocation::GPU), bias({1, channels}, TensorLocation::GPU), saveMean({1, channels}, TensorLocation::GPU), saveVariance({1, channels}, TensorLocation::GPU), runMean({1, channels}, TensorLocation::GPU), runVariance({1, channels}, TensorLocation::GPU), biasGrad({1, channels}, TensorLocation::GPU), scaleGrad({1, channels}, TensorLocation::GPU), epsilon(1e-5f){
  if(conv){
    this->mode = CUDNN_BATCHNORM_SPATIAL;
  }
  else{
    this->mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  }

  TryCuda(cudnnCreateTensorDescriptor(&inpD));
  TryCuda(cudnnCreateTensorDescriptor(&memD));
  TryCuda(cudnnSetTensor4dDescriptor(memD, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, channels, 1, 1));
}

NormalizationLayer::NormalizationLayer(const NormalizationLayer& lay) : scale(lay.scale), bias(lay.bias), saveMean(lay.saveMean), saveVariance(lay.saveVariance), runMean(lay.runMean), runVariance(lay.runVariance), biasGrad(lay.biasGrad.dimensions, TensorLocation::GPU, lay.biasGrad.n), scaleGrad(lay.scaleGrad.dimensions, TensorLocation::GPU, lay.scaleGrad.n), mode(lay.mode){
  TryCuda(cudnnCreateTensorDescriptor(&this->inpD));
  TryCuda(cudnnCreateTensorDescriptor(&this->memD));
  TryCuda(cudnnSetTensor4dDescriptor(this->memD, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, this->bias.dimensions[1], 1, 1));
  this->epsilon = lay.epsilon;
}

NormalizationLayer::~NormalizationLayer(){}

std::unique_ptr<Layer> NormalizationLayer::clone(){
  return(std::make_unique<NormalizationLayer>(*this));
}

Tensor NormalizationLayer::forward(Tensor<__half>& T, bool train){
  auto start = std::chrono::steady_clock::now();
  TryCuda(cudnnSetTensor4dDescriptor(this->inpD, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], T.dimensions[2], T.dimensions[3]));
  
  if(train){
    this->input = T;
    TryCuda(cudnnBatchNormalizationForwardTraining(nnHandle, this->mode, &mx, &mn, this->inpD, T.gpuData(), this->inpD, T.gpuData(), this->memD, this->scale.gpuData(), this->bias.gpuData(), 
                                                  0.1, this->runMean.gpuData(), this->runVariance.gpuData(), this->epsilon, this->saveMean.gpuData(), this->saveVariance.gpuData()));
  }
  else{
    TryCuda(cudnnBatchNormalizationForwardInference(nnHandle, this->mode, &mx, &mn, this->inpD, T.gpuData(), this->inpD, T.gpuData(), this->memD, this->scale.gpuData(), this->bias.gpuData(), 
                                                    this->runMean.gpuData(), this->runVariance.gpuData(), this->epsilon));
  }
  auto elapsed = std::chrono::steady_clock::now() - start;
  ////std::cout<<std::string("Time in normalization: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  return std::move(T);
}
Tensor NormalizationLayer::backward(Tensor<__half>& gradient){
  auto start = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::steady_clock::now() - start;
  TryCuda(cudnnBatchNormalizationBackward(nnHandle, this->mode, &mx, &mn, &mx, &mn, this->inpD, this->input.gpuData(), this->inpD, gradient.gpuData(), this->inpD, gradient.gpuData(), this->memD,
                                          this->scale.gpuData(), this->scaleGrad.gpuData(), this->biasGrad.gpuData(), this->epsilon, this->saveMean.gpuData(), this->saveVariance.gpuData()));
  
  //TryCuda(cudaDeviceSynchronize());
  elapsed = std::chrono::steady_clock::now() - start;
  //std::cout<<std::string("Time in normalization back: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  
  return std::move(gradient);
}

void NormalizationLayer::genTensorData(){
  this->scale.gpuSend();
  this->bias.gpuSend();
  Generator::tGen(this->scale);
  //Generator::vGen(this->bias.size, 0, this->bias.gpuData());
  //Generator::vGen(this->runMean.size, 0, this->runMean.gpuData());
  Generator::vGen(this->runVariance.size, 1, this->runVariance.gpuData());  
}

void NormalizationLayer::loadTensor(std::ifstream& iF){
  this->scale.readBinary(iF);
  this->bias.readBinary(iF);
  this->runMean.readBinary(iF);
  this->runVariance.readBinary(iF);
  this->saveMean.readBinary(iF);
  this->saveVariance.readBinary(iF);
  this->scale.gpuSend();
  this->bias.gpuSend();
  this->runMean.gpuSend();
  this->runVariance.gpuSend();
  this->saveMean.gpuSend();
  this->saveVariance.gpuSend();
}

void NormalizationLayer::saveTensor(std::ofstream& oF){
  this->scale.writeBinary(oF);
  this->bias.writeBinary(oF);
  this->runMean.writeBinary(oF);
  this->runVariance.writeBinary(oF);
  this->saveMean.writeBinary(oF);
  this->saveVariance.writeBinary(oF);
  this->scale.gpuSend();
  this->bias.gpuSend();
  this->runMean.gpuSend();
  this->runVariance.gpuSend();
  this->saveMean.gpuSend();
  this->saveVariance.gpuSend();
}

void NormalizationLayer::cleanSave(std::ofstream& oF){
  scale.writeTensor(oF);
  bias.writeTensor(oF);
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> NormalizationLayer::getLearningData(){
  return std::make_pair(std::vector<Tensor*>{&this->scale, &this->bias}, std::vector<Tensor*>{&this->scaleGrad, &this->biasGrad});
  scale.gpuSend();
  bias.gpuSend();
}