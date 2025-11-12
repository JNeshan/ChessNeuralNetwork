//denseLayer.h
#ifndef DENSELAYER_H
#define DENSELAYER_H
#include "layer.h"

class DenseLayer : public Layer {
public:
  DenseLayer(const int f, const int n);
  DenseLayer(const DenseLayer& lay);
  virtual ~DenseLayer();
  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) override;
  virtual BackwardPackage backward(Tensor<__half>& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  
  Tensor<__half> input, weight, bias, wGrad, bGrad;


private:
};

#endif