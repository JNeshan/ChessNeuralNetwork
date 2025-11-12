#ifndef MATRICEMATH_H
#define MATRICEMATH_H

#include <iostream>
#include "cuda_runtime.h"
#include "cudnn.h"
#include "cublas.h"
#include "tensor.cuh"
class MatriceMath{
public:
  MatriceMath();
  void add(const Tensor<__half>& N, const Tensor<__half>& B, Tensor<__half>& O);
  void multiply(const Tensor<__half>& M, const Tensor<__half>& N, Tensor<__half>& O);
  void convolution(const Tensor<__half>& M, const Tensor<__half>& F, Tensor<__half>& O);
  cudnnHandle_t cudnn;
  cublasHandle_t cublas;
};

#endif