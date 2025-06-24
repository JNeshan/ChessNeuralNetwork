#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"

struct CudaMembers;

class DenseLayer : public Layer {
public:
  DenseLayer(const int f, const int n);
  virtual ~DenseLayer();
  virtual Tensor forward(const Tensor& T) override;
  void loadParameters(std::ifstream iF);

private:
  Tensor weight, bias;
  CudaMembers *CudaM;
};

#endif