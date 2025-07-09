#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

struct CudaMembers;

class ReLULayer : public Layer{
public:
  ReLULayer();
  ~ReLULayer();
  virtual std::pair<Tensor, std::unique_ptr<ForwardCache>> forward(const Tensor& T) override;
  virtual std::pair<Tensor, std::unique_ptr<BackwardCache>> backward(const Tensor& gradient, const ForwardCache& fCache) override;
  CudaMembers *CudaM;

private:
};

#endif