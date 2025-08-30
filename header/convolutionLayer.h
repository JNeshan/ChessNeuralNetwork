//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"


class ConvolutionLayer : public Layer {
public:
  ConvolutionLayer(const int fC, const int iC, const int fH, const int fW);
  ~ConvolutionLayer();
  virtual Tensor forward(const Tensor& T, bool train) override;
  virtual Tensor backward(const Tensor& gradient) override;

  bool y; //mutex for training, only one 

private:
  int filterSize;
  Tensor input, filters, bias, iGrad, fGrad, bGrad; //filter and bias tensors
  cudnnFilterDescriptor_t filterD; 
  cudnnTensorDescriptor_t inputD, biasD, outputD;
  cudnnConvolutionDescriptor_t convoD;
};

#endif