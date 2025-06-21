//tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>

enum class TensorLocation {CPU, GPU};

class Tensor {
public:
    Tensor(const std::vector<int>& dim);
    Tensor(const std::vector<int>& dim, const std::vector<float>& data);
    ~Tensor();

    void gpuSend(); //moves data into devices memory
    void cpuSend();

    float* cpuData(); //retrieves pointer to data
    float* gpuData();

    int size;

private:
    std::vector<int> dimensions;
    std::vector<float> tensor;
    TensorLocation device;
    float* data;
    
};

#endif