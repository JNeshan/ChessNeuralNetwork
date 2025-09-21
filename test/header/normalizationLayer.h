//normalizationLayer.h
#ifndef NORMALIZATIONLAYER_H
#define NORMALIZATIONLAYER_H
#include "layer.h"

class NormalizationLayer : public Layer{
public:
  NormalizationLayer(bool conv, int channels);
  NormalizationLayer(const NormalizationLayer& lay);
  virtual ~NormalizationLayer();
  virtual Tensor forward(Tensor& T, bool train) override;
  virtual Tensor backward(Tensor& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  virtual std::pair<std::vector<Tensor*>, std::vector<Tensor*>> getLearningData() override;
private:
  Tensor scale, bias, runMean, runVariance, saveVariance, saveMean, input, biasGrad, scaleGrad;
  cudnnBatchNormMode_t mode;
  cudnnTensorDescriptor_t inpD, memD;
  float epsilon;
};

#endif