//softmaxLayer.h
#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "layer.h"

struct CudaMembers;

class SoftmaxLayer : Layer{
public:
  SoftmaxLayer();
  ~SoftmaxLayer();

  virtual std::pair<Tensor, std::unique_ptr<ForwardCache>> forward(const Tensor& T) override;
  virtual std::pair<Tensor, std::unique_ptr<BackwardCache>> backward(const Tensor& gradient, const ForwardCache& fCache) override;

private:
  CudaMembers *CudaM;
  Tensor output;
  std::vector<int> dimensions;
  const int outFeat;
  int n;
};


#endif