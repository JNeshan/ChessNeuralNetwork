//tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>

enum class TensorLocation {CPU, GPU};

class Tensor {
public:
    Tensor(); //blank constructor
    Tensor(const std::vector<int>& dim, const TensorLocation loc = TensorLocation::CPU);
    Tensor(Tensor&& r) noexcept;
    Tensor& operator=(Tensor&& r) noexcept;
    ~Tensor();
    Tensor& operator=(const Tensor& r);
    Tensor(const Tensor& r);

    void gpuSend(); //moves data into devices memory
    void cpuSend();
    void flatten();
    void reshape(const std::vector<int>& dim);

    float* cpuData() const; //retrieves pointer to data
    float* gpuData() const;

    void metadata(); //used to output and check data stored in tensor

    //public member variables
    int size;
    std::vector<int> dimensions;
    //const int dSize;



private:
    TensorLocation device;
    float* data;
    
};

#endif