//normalizationLayer.h
#ifndef NORMALIZATIONLAYER_H
#define NORMALIZATIONLAYER_H
#include "layer.h"

class NormalizationLayer : public Layer{
public:
  NormalizationLayer(bool conv, int channels);
  NormalizationLayer(const NormalizationLayer& lay);
  virtual ~NormalizationLayer();
  virtual Tensor<__half> forward(Tensor<__half>& T, bool train) override;
  virtual BackwardPackage backward(Tensor<__half>& gradient) override;
  virtual void genTensorData() override;
  virtual void loadTensor(std::ifstream& iF) override;
  virtual void saveTensor(std::ofstream& oF) override;
  virtual void cleanSave(std::ofstream& oF) override; 

  virtual std::unique_ptr<Layer> clone() override;
  
private:
  Tensor<__half> scale, bias, input, biasGrad, scaleGrad;
  Tensor<float> runMean, runVariance, saveVariance, saveMean;
  cudnnBatchNormMode_t mode;
  cudnnTensorDescriptor_t inpD, memD;
  float epsilon;
};

#endif