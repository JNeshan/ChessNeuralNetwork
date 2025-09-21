#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H
#include "layer.h"

class FlattenLayer : public Layer {
public:
  FlattenLayer();
  FlattenLayer(const FlattenLayer& lay);
  virtual ~FlattenLayer();
  virtual Tensor forward(Tensor& T, bool train) override;
  virtual Tensor backward(Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;
private:
  std::vector<int> inpDim;
};

#endif