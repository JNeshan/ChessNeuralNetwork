//optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

class Optimizer{
public:
  Optimizer(const float rate);
  ~Optimizer();
  void optimize(const Tensor& in, const Tensor& grad);
  static void setRate(const float rate);
  const float lR;
private:
  

};

#endif