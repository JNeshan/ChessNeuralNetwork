#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"

class DenseLayer : public Layer {
public:
  DenseLayer(const int f, const int n);
  virtual ~DenseLayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual Tensor backward(const Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override;

private:
  Tensor input, weight, bias, wGrad, bGrad;
};

#endif