#ifndef RESIDUALBLOCK_H
#define RESIDUALBLOCK_H
#include "layer.h"
#include "convolutionLayer.h"
#include "denseLayer.h"
#include "reluLayer.h"
#include "tanhLayer.h"
#include "softmaxLayer.h"
#include "neuralNetwork.h"
#include "normalizationLayer.h"
#include <memory>

//effectively a layer that controls layers to enforce the skip connection logic

class ResidualBlock : public Layer{
public:
  ResidualBlock();
  ResidualBlock(const ResidualBlock& lay);
  virtual ~ResidualBlock();
  virtual std::unique_ptr<Layer> clone() override;
  //ResidualBlock(const int channels, const int filterSize, const int filterCount);
  //void appendLayers(NeuralNetwork& net);
  virtual Tensor forward(Tensor& T, bool train) override;
  virtual Tensor backward(Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF);
  virtual void saveTensor(std::ofstream& oF);
  virtual void cleanSave(std::ofstream& oF);
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;
private:
  std::vector<std::unique_ptr<Layer>> layers;
  Tensor inp, inpGrad;
};

#endif