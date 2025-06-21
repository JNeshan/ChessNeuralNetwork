#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <memory>
#include <fstream>
#include <stdexcept>
#include <vector>

struct CudaMembers;

class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual void forward(Tensor T) = 0;
    virtual void backward() = 0;
    
private:
  
};

#endif // LAYER_H