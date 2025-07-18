#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

struct CudaMembers;

class ReLULayer : public Layer{
public:
  ReLULayer();
  ~ReLULayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual Tensor backward(const Tensor& gradient) override;
private:
  Tensor input;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t reLU;
};

#endif