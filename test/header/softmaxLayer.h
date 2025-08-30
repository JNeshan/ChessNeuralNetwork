//softmaxLayer.h
#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "layer.h"

struct CudaMembers;

class SoftmaxLayer : Layer{
public:
  SoftmaxLayer();
  ~SoftmaxLayer();

  virtual Tensor forward(const Tensor& T, bool train) override;
  virtual Tensor backward(const Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;

private:
  Tensor output;
  std::vector<int> dimensions;
  const int outFeat;
  int n;
  cudnnTensorDescriptor_t tensorD;
};


#endif