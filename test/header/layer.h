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
#include <chrono>
#include "generator.h"

class Layer {
public:
  Layer();
  Layer(const Layer& lay);
  virtual ~Layer();
  virtual std::unique_ptr<Layer> clone() = 0;
  virtual Tensor forward(Tensor& T, bool train) = 0;
  virtual Tensor backward(Tensor& gradient) = 0;

  const float mx = 1.0f, mn = 0.0f;
  virtual void genTensorData() = 0;
  virtual void loadTensor(std::ifstream& iF) = 0;
  virtual void saveTensor(std::ofstream& oF) = 0;
  virtual void cleanSave(std::ofstream& oF) = 0;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData()= 0;
  static thread_local cudnnHandle_t nnHandle;
  static thread_local cublasHandle_t blasHandle;
};

#endif 