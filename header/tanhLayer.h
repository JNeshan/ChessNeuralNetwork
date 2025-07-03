#ifndef TANHLAYER_H
#define TANHLAYER_H

#include "layer.h"


class tanhLayer : Layer {
public:
  tanhLayer();
  virtual ~tanhLayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> backward(const Tensor& gradient) override;

private:
  CudaMembers *CudaM;
  Tensor input, output;
};

#endif