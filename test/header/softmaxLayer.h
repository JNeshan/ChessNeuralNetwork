//softmaxLayer.h
#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "layer.h"

class SoftmaxLayer : public Layer{
public:
  SoftmaxLayer();
  SoftmaxLayer(const SoftmaxLayer& lay);
  virtual ~SoftmaxLayer();


  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) override;
  virtual BackwardPackage backward(Tensor<__half>& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  

private:
  Tensor<__half> output;
  std::vector<int> dimensions;
  const int outFeat;
  int n;
  cudnnTensorDescriptor_t tensorD;
};


#endif