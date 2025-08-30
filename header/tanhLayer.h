#ifndef TANHLAYER_H
#define TANHLAYER_H

#include "layer.h"


class tanhLayer : public Layer {
public:
  tanhLayer();
  virtual ~tanhLayer();
  virtual Tensor forward(const Tensor& T, bool train) override;
  virtual Tensor backward(const Tensor& gradient) override;

private:
  Tensor output;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t actD;
};

#endif