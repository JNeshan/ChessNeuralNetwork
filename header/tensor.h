//tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>

enum class TensorLocation {CPU, GPU};
//The tensor class wraps a pointer to a contiguous array with data to represent it as an n-dimensional matrix, along with
//handling the allocation, destruction, and transfer of memory in the CPU and GPU
//The dimensions and overall size are necessary to perform the Cuda computations, and the dimension count n is needed to
//enforce constraints for a layer that expects an input tensor with specific dimensions (i.e. convolution using 4, dense using 2)
//Reshape can freely modify dimensions and n as long as size remains the same, since its encapsulating the data but with a different
//split.
//NOTE: Tensor will always pad dimensions to be at least size 4. This is essential since cudnn tensor descriptors require 4
//dimension inputs, so by padding dimensions with 1, which keeps size the same, and using n to recall the actual dimensionality,
//the descriptor calls can always use the four indices of dimensions. This just makes it much simpler.
//
//Current code does not adhere to the current tensor rules. The following must be enforced:
//dimensions.size will not always relate to the dimensionality of the tensor. The only time it is assumed to is when no
//dimensionality is passed in the constructor. If a tensor object is created and uses the default nth, then the passed dim
//is assumed to be unpadded. A dimension size of 1 for a specific dimension does not guarentee that it is a padded dimension,
//so there is no way for the constructor to ensure that dim is the right size in the default case.
//This should be a non-issue though, since a padded vector could only ever come from another tensor, in which case its n 
//can be passed to remedy this

class Tensor {
public:
    Tensor(); //blank constructor
    Tensor(const std::vector<int>& dim, const TensorLocation loc, const int nth = 0);
    Tensor(const Tensor& r); //deep copy constructor
    Tensor(Tensor&& r); //move constructor
    ~Tensor(); //destructor
    Tensor& operator=(Tensor&& r); //i don't remember what the equal ones are called
    Tensor& operator=(const Tensor& r); 

    
    //modifies the shape of the matrix to a new set of dimensions that must be the same size as the previous, only modifies dimensions and n
    void reshape(const std::vector<int>& dim, const int nth); 

    //moves data into respective devices memory
    void gpuSend(); 
    void cpuSend();

    //retrieves pointer to data from respective device
    float* cpuData() const; 
    float* gpuData() const;

    //public member variables
    int size, n; //total number of elements and the highest order dimension
    std::vector<int> dimensions; //dimension sizes

private:
    TensorLocation device;
    float* data;
};

#endif