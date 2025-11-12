#ifndef TANHLAYER_H
#define TANHLAYER_H

#include "layer.h"


class tanhLayer : public Layer {
public:
  tanhLayer();
  tanhLayer(const tanhLayer& lay);
  virtual ~tanhLayer();
  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) override;
  virtual BackwardPackage backward(Tensor<__half>& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  

private:
  Tensor<__half> output;
  cudnnTensorDescriptor_t tensorD;
  cudnnActivationDescriptor_t actD;
};

#endif