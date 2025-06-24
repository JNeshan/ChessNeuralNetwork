#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

struct CudaMembers;

class ReLULayer : public Layer{
public:
  ReLULayer();
  virtual ~ReLULayer();
  Tensor forward(const Tensor& T) override;
  void backward() override;
  CudaMembers *CudaM;
private:
};

#endif