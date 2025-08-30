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
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;

  bool y; //mutex for training, only one 
  Tensor input, filters, bias, iGrad, fGrad, bGrad; //filter and bias tensors


private:
  int filterSize;
  cudnnFilterDescriptor_t filterD; 
  cudnnTensorDescriptor_t inputD, biasD, outputD;
  cudnnConvolutionDescriptor_t convoD;
};

#endif