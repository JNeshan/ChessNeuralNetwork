#ifndef MATRICEMATH_H
#define MATRICEMATH_H

#include <iostream>
#include "cuda_runtime.h"
#include "cudnn.h"
#include "cublas.h"
#include "tensor.h"

class MatriceMath{
public:
  MatriceMath();
  void add(const Tensor& N, const Tensor& B, Tensor& O);
  void multiply(const Tensor& M, const Tensor& N, Tensor& O);
  void convolution(const Tensor& M, const Tensor& F, Tensor& O);
  cudnnHandle_t cudnn;
  cublasHandle_t cublas;
};

#endif