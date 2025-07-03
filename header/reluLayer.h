#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

struct CudaMembers;

class ReLULayer : public Layer{
public:
  ReLULayer();
  ~ReLULayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> backward(const Tensor& gradient) override;
  CudaMembers *CudaM;

private:
  Tensor input, iGrad;

};

#endif