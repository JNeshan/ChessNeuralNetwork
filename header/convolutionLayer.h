//convolutionLayer.h
#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H
#include "layer.h"

struct CudaMembers;


class ConvolutionLayer : Layer {
public:
  ConvolutionLayer(const int fC, const int iC, const int fH, const int fW);
  ~ConvolutionLayer();
  virtual Tensor forward(const Tensor& T) override;
  virtual void backward() override;
  void loadParameters(std::ifstream& iF);
  void saveParameters(std::ofstream& oF);

private:
  int filterSize;
  Tensor output, filters, bias; //filter and bias tensors
  CudaMembers *CudaM; //holds cuda member variables

};

#endif