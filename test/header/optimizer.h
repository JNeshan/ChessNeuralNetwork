//optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.cuh"
class Optimizer{
public:
  Optimizer(const float rate);
  ~Optimizer();
  void optimize(const Tensor<float>& in, const Tensor<__half>& grad);
  void batchOptimize(std::pair<std::vector<Tensor<float>*>, std::vector<Tensor<__half>*>>& trainingBatch);
  void setRate(const float rate);
  float lR;
private:
  

};

#endif