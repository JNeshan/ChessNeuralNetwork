//softmaxLayer.h
#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "layer.h"

struct CudaMembers;

class SoftmaxLayer : Layer{
public:
  SoftmaxLayer();
  ~SoftmaxLayer();

  virtual Tensor forward(const Tensor& T) override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> backward(const Tensor& gradient) override;

private:
  CudaMembers *CudaM;
  Tensor input, output, iGrad, oGrad;
  std::vector<int> dimensions;
  const int outFeat;
};


#endif