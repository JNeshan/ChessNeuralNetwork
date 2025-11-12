//layer.h
#ifndef LAYER_H
#define LAYER_H

#include "tensor.cuh"
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

struct ForwardPackage{
  ForwardPackage(Tensor<__half>& o, std::vector<Tensor<__half>>& error);
  ForwardPackage(Tensor<__half>& o);

  Tensor<__half> output;
  std::vector<Tensor<__half>> errSet;
};

struct BackwardPackage{
  BackwardPackage(Tensor<__half>& o, std::vector<Tensor<__half>>& gs);
  BackwardPackage(Tensor<__half>& o);
  Tensor<__half> iGrad;
  std::vector<Tensor<__half>> trainGrads;
};

class Layer{
public:
  Layer();
  Layer(const Layer& lay);
  virtual ~Layer();
  virtual std::unique_ptr<Layer> clone() = 0;
  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) = 0;
  virtual BackwardPackage backward(Tensor<__half>& gradient) = 0;

  const __half mx = 1.0f, mn = 0.0f;
  virtual void genTensorData() = 0;
  virtual void loadTensor(std::ifstream& iF) = 0;
  virtual void saveTensor(std::ofstream& oF) = 0;
  virtual void cleanSave(std::ofstream& oF) = 0;
  static void* wsPtr;
  static size_t wsSize;
  static thread_local cudnnHandle_t nnHandle;
  static thread_local cublasHandle_t blasHandle;
};

#endif 