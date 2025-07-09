#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"

struct CudaMembers;

class DenseLayer : public Layer {
public:
  DenseLayer(const int f, const int n);
  virtual ~DenseLayer();
  virtual std::pair<Tensor, std::unique_ptr<ForwardCache>> forward(const Tensor& T) override;
  virtual std::pair<Tensor, std::unique_ptr<BackwardCache>> backward(const Tensor& gradient, const ForwardCache& fCache) override;
  void loadParameters(std::ifstream iF);

private:
  Tensor weight, bias, wGrad, bGrad;
  CudaMembers *CudaM;
};

#endif