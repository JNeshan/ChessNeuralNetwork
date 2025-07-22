#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

struct CudaMembers;

class ReLULayer : public Layer{
public:
  ReLULayer();
  ~ReLULayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual Tensor backward(const Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override;
  
  Tensor input;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t reLU;

private:
  
};

#endif