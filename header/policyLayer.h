//policyLayer.h
#ifndef POLICYLAYER_H
#define POLICYLAYER_H

#include "layer.h"

struct CudaMembers;

class PolicyLayer : Layer{
public:
  PolicyLayer();
  ~PolicyLayer();

  virtual Tensor forward(const Tensor& T) override;
  virtual void backward() override;



private:
  CudaMembers *CudaM;
  
};


#endif