//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"

struct CudaMembers;


class ConvolutionLayer : Layer {
public:
  ConvolutionLayer(const int fC, const int iC, const int fH, const int fW);
  ~ConvolutionLayer();
  virtual std::pair<Tensor, std::unique_ptr<ForwardCache>> forward(const Tensor& T) override;
  virtual std::pair<Tensor, std::unique_ptr<BackwardCache>> backward(const Tensor& gradient, const ForwardCache& fCache) override;
  void loadParameters(std::ifstream& iF);
  void saveParameters(std::ofstream& oF);

private:
  int filterSize;
  Tensor filters, bias; //filter and bias tensors
  CudaMembers *CudaM; //holds cuda member variables
};

#endif