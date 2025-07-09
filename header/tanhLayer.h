#ifndef TANHLAYER_H
#define TANHLAYER_H

#include "layer.h"


class tanhLayer : Layer {
public:
  tanhLayer();
  virtual ~tanhLayer();
  virtual std::pair<Tensor, std::unique_ptr<ForwardCache>> forward(const Tensor& T) override;
  virtual std::pair<Tensor, std::unique_ptr<BackwardCache>> backward(const Tensor& gradient, const ForwardCache& fCache) override;

private:
  CudaMembers *CudaM;
  Tensor output;
};

#endif