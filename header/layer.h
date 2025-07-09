//layer.h
#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <utility>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <vector>

//
//All layers are currently working under the assumption that only a single forward pass can be performed at a time
//which will not work in parallel. A layer object needs to be able to perform any number of forward operations while
//still allowing access to each ones tensor details needed for backpropagation. The current issue is due to storing the
//descriptors and 
//
//
//

struct CudaMembers; //each layer subclass defines a unique version that only it knows how to work with
//cache structs used to send data back to the caller to hold onto so it can be used for backpropagation or training
struct ForwardCache{ 
public:
    ForwardCache(const Tensor& tensor, const CudaMembers& c);
    CudaMembers* CudaM;
    Tensor T;
};

struct BackwardCache{
public:
    BackwardCache();
    std::vector<std::pair<Tensor*, Tensor>> trainingTensors;
    void cachePair(const Tensor& m, const Tensor& grad);
};

class Layer {
public:
    Layer();
    virtual ~Layer();
    virtual std::pair<Tensor, std::unique_ptr<BackwardCache>> backward(const Tensor& gradient, const ForwardCache& fCache) = 0;
    virtual std::pair<Tensor, std::unique_ptr<ForwardCache>> forward(const Tensor& T) = 0;
    static int alpha, beta;
    const float mx = 1.0f, mn = 0.0f;
    void genTensorData();
    void loadTensor(std::ifstream& iF);
};

#endif 