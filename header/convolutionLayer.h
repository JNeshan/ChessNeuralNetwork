//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"

struct CudaMembers;


class ConvolutionLayer : Layer {
public:
  ConvolutionLayer(const int fC, const int iC, const int fH, const int fW);
  ~ConvolutionLayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> backward(const Tensor& gradient) override;
  void loadParameters(std::ifstream& iF);
  void saveParameters(std::ofstream& oF);

private:
  int filterSize;
  std::pair<std::vector<Tensor*>, std::vector<Tensor*>> tensors;
  Tensor output, filters, bias, iGrad, fGrad, bGrad; //filter and bias tensors
  CudaMembers *CudaM; //holds cuda member variables
};

#endif