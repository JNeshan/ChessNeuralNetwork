#ifndef RELULAYER_H
#define  RELULAYER_H
#include "layer.h"



class ReLULayer : public Layer{
public:
  ReLULayer();
  virtual ~ReLULayer();
  void forward(Tensor T) override;
  void backward() override;
};

#endif