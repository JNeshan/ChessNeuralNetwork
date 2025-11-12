#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"

class ReLULayer : public Layer{
public:
  ReLULayer();
  ReLULayer(const ReLULayer& lay);
  virtual ~ReLULayer();
  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) override;
  virtual BackwardPackage backward(Tensor<__half>& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  

  
  Tensor<__half> input;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t reLU;

private:
  
};

#endif