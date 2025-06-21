#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"

class DenseLayer : public Layer {
public:
  DenseLayer();
  virtual ~DenseLayer();
  virtual void forward(Tensor T) override;
  void loadParameters(std::ifstream iF);

private:
  Tensor weight, bias;
};

#endif