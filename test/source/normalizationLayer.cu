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

NormalizationLayer::NormalizationLayer(bool conv, int channels) : scale({1, channels}, TensorLocation::GPU), bias({1, channels}, TensorLocation::GPU), saveMean({1, channels}, TensorLocation::GPU), saveVariance({1, channels}, TensorLocation::GPU), runMean({1, channels}, TensorLocation::GPU), runVariance({1, channels}, TensorLocation::GPU), biasGrad({1, channels}, TensorLocation::GPU), scaleGrad({1, channels}, TensorLocation::GPU){
  if(conv){
    mode = CUDNN_BATCHNORM_SPATIAL;
  }
  else{
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  }

  TryCuda(cudnnCreateTensorDescriptor(&inpD));
  TryCuda(cudnnCreateTensorDescriptor(&memD));
  TryCuda(cudnnSetTensor4dDescriptor(memD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channels, 1, 1));
}


Tensor NormalizationLayer::forward(const Tensor& T, bool train){
  this->input = T;
  Tensor output(T.dimensions, TensorLocation::GPU, T.n);
  TryCuda(cudnnSetTensor4dDescriptor(this->inpD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, T.dimensions[0], T.dimensions[1], T.dimensions[2], T.dimensions[3]));

  if(train){
    TryCuda(cudnnBatchNormalizationForwardTraining(nnHandle, this->mode, &mx, &mn, this->inpD, T.gpuData(), this->inpD, output.gpuData(), this->memD, this->scale.gpuData(), 
                                          this->bias.gpuData(), 1.0, this->runMean.gpuData(), this->runVariance.gpuData(), this->epsilon, this->saveMean.gpuData(), this->saveVariance.gpuData()));
  }
  else{
    TryCuda(cudnnBatchNormalizationForwardInference(nnHandle, this->mode, &mx, &mn, this->inpD, T.gpuData(), this->inpD, output.gpuData(), this->memD, this->scale.gpuData(), this->bias.gpuData(), this->runMean.gpuData(), this->runVariance.gpuData(), this->epsilon));
  }
  return output;
}
Tensor NormalizationLayer::backward(const Tensor& gradient){
  Tensor inputGrad(input.dimensions, TensorLocation::GPU);
  TryCuda(cudnnBatchNormalizationBackward(nnHandle, this->mode, &mx, &mn, &mx, &mn, this->inpD, this->input.gpuData(), this->inpD, gradient.gpuData(), this->inpD, inputGrad.gpuData(), this->memD,
                                          this->scale.gpuData(), this->scaleGrad.gpuData(), this->biasGrad.gpuData(), this->epsilon, this->saveMean.gpuData(), this->saveVariance.gpuData()));
  return inputGrad;
}

void NormalizationLayer::genTensorData(){
  Generator::tGen(scale);
  Generator::vGen(bias.size, 0, bias.gpuData());
}
void NormalizationLayer::loadTensor(std::ifstream& iF){
  scale.readBinary(iF);
  bias.readBinary(iF);
}
void NormalizationLayer::saveTensor(std::ofstream& oF){
  scale.writeBinary(oF);
  bias.writeBinary(oF);
}
void NormalizationLayer::cleanSave(std::ofstream& oF){
  scale.writeTensor(oF);
  bias.writeTensor(oF);
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> NormalizationLayer::getLearningData(){
  return std::make_pair(std::vector<Tensor*>{&this->scale, &this->bias}, std::vector<Tensor*>{&this->scaleGrad, &this->biasGrad});
}