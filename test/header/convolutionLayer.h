//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"


class ConvolutionLayer : public Layer {
public:
  ConvolutionLayer(const int fC, const int fH, const int fW, const int iC, const int pad);
  ConvolutionLayer(const ConvolutionLayer& lay);
  virtual ~ConvolutionLayer();
  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) override;
  virtual BackwardPackage backward(Tensor<__half>& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 
  virtual std::unique_ptr<Layer> clone() override;
  

  Tensor<__half> input, filters, bias, iGrad, fGrad, bGrad; //filter and bias tensors


private:
  const int padding;
  int n, h, w, c; //
  bool forw, back;
  //size_t wsSizeF, wsSizeB; remove for global workspace
  //void* wsPtrF;
  //void* wsPtrB;
  cudnnFilterDescriptor_t filterD; 
  cudnnTensorDescriptor_t inputD, biasD, outputD;
  cudnnActivationDescriptor_t actD;
  cudnnConvolutionDescriptor_t convoD;
  cudnnConvolutionFwdAlgo_t convoAlgo;
  cudnnConvolutionBwdFilterAlgo_t backFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t backDataAlgo;
};

#endif