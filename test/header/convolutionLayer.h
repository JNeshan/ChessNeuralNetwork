//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"


class ConvolutionLayer : public Layer {
public:
  ConvolutionLayer(const int fC, const int iC, const int fH, const int fW, const int pad);
  ConvolutionLayer(const ConvolutionLayer& lay);
  virtual ~ConvolutionLayer();
  virtual Tensor forward(Tensor& T, bool train) override;
  virtual Tensor backward(Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 
  virtual std::unique_ptr<Layer> clone() override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;

  Tensor input, filters, bias, iGrad, fGrad, bGrad; //filter and bias tensors


private:
  const int padding;
  bool forw, back;
  size_t wsSizeF, wsSizeB;
  void* wsPtrF;
  void* wsPtrB;
  cudnnFilterDescriptor_t filterD; 
  cudnnTensorDescriptor_t inputD, biasD, outputD;
  cudnnActivationDescriptor_t actD;
  cudnnConvolutionDescriptor_t convoD;
  cudnnConvolutionFwdAlgo_t convoAlgo;
  cudnnConvolutionBwdFilterAlgo_t backFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t backDataAlgo;
};

#endif