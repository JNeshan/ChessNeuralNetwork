#ifndef TANHLAYER_H
#define TANHLAYER_H

#include "layer.h"


class tanhLayer : Layer {
public:
  tanhLayer();
  virtual ~tanhLayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual Tensor backward(const Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override;

private:
  Tensor output;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t actD;
};

#endif