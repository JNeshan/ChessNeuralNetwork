#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <memory>
#include <fstream>
#include <stdexcept>
#include <vector>


class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual Tensor forward(const Tensor& T) = 0;
    virtual void backward() = 0;
    static int alpha, beta;
    Tensor input;
    const float mx = 1.0f, mn = 0.0f;
    void genTensorData();
    void loadTensor(std::ifstream& iF);


};

#endif // LAYER_H