//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"

class ConvolutionLayer : Layer {
public:
  ConvolutionLayer(std::vector<int>& dim, int fCount, int fSize);
  ~ConvolutionLayer();
  virtual void forward(Tensor T) override;
  virtual void backward() override;
  void loadParameters(std::ifstream& iF);
  void saveParameters(std::ofstream& oF);

private:
  int filterSize;
  Tensor filters, bias; //filter and bias tensors
  CudaMembers CudaM; //holds cuda member variables
};

#endif