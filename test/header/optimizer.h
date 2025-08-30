//optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

class Optimizer{
public:
  Optimizer(const float rate);
  ~Optimizer();
  void optimize(const Tensor& in, const Tensor& grad);
  void batchOptimize(std::pair<std::vector<Tensor*>, std::vector<Tensor*>>& trainingBatch);
  void setRate(const float rate);
  float lR;
private:
  

};

#endif