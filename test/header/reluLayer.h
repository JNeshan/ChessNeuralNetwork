#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

class ReLULayer : public Layer{
public:
  ReLULayer();
  ReLULayer(const ReLULayer& lay);
  virtual ~ReLULayer();
  virtual Tensor forward(Tensor& T, bool train) override;
  virtual Tensor backward(Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;

  
  Tensor input;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t reLU;

private:
  
};

#endif