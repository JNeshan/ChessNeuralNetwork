#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"

class DenseLayer : public Layer {
public:
  DenseLayer(const int f, const int n);
  virtual ~DenseLayer();
  virtual Tensor forward(const Tensor& T, bool train) override;
  virtual Tensor backward(const Tensor& gradient) override;
  void loadParameters(std::ifstream iF);

private:
  Tensor input, weight, bias, wGrad, bGrad;
};

#endif