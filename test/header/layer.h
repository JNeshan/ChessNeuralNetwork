//layer.h
#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "cublas.h"
#include <utility>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "generator.h"

class Layer {
public:
  Layer();
  //Layer(const Layer& r);
  virtual ~Layer();
  virtual Tensor forward(const Tensor& T) = 0;
  virtual Tensor backward(const Tensor& gradient) = 0;

  const float mx = 1.0f, mn = 0.0f;
  virtual void genTensorData() = 0;
  virtual void loadTensor(std::ifstream& iF) = 0;
  virtual void saveTensor(std::ofstream& oF) = 0;
  virtual void cleanSave(std::ofstream& oF) = 0;
  static thread_local cudnnHandle_t nnHandle;
  static thread_local cublasHandle_t blasHandle;
};

#endif 