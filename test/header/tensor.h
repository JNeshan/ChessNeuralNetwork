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
//Tensors have their data including their sizes, dimensionality, and array data stored into binary files. They are written to the immediately
//open section so sequential calls to write to the same open file will write them sequentially. As such the read function expects to read
//tensor information from an input file in a very strict order, with it using the data type sizes to move across the data and information like
//size to know how far to extend the float*. As such, the order that the tensors are saved in needs to be maintained across runs, so for the
//overall neural network, a modification to the layers in any form means that the data either needs to be reset, or there needs to be a function
//specifically made to change the structure of layers and maintain the old layers data.
//The main draw from this is that writing and reading must be performed very carefully.
//

//tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>

enum class TensorLocation {CPU, GPU};

//

class Tensor {
public:
    Tensor(); //blank constructor
    Tensor(const std::vector<int>& dim, const TensorLocation loc = TensorLocation::CPU, int nth = 0); //primary constructor
    
    Tensor(const Tensor& r); //deep copy constructor
    Tensor(std::vector<Tensor>& t);
    Tensor(Tensor&& r); //move constructor
    ~Tensor(); //destructor
    Tensor& operator=(Tensor&& r); //i don't remember what the equal ones are called
    Tensor& operator=(const Tensor& r);
    void batch(Tensor& B);
    void gpuAdd(Tensor& B);
    void batchBuild(const std::vector<Tensor*>& t, const TensorLocation loc = TensorLocation::CPU);
    void batchBuild(const std::vector<Tensor>& t, const TensorLocation loc = TensorLocation::CPU);
    Tensor segment(const int n_i);

    
    //modifies the shape of the matrix to a new set of dimensions that must be the same size as the previous, only modifies dimensions and n
    void reshape(const std::vector<int>& dim, const int nth); 

    //moves data into respective devices memory
    void gpuSend(); 
    void cpuSend();

    //retrieves pointer to data from respective device

    //gives pointer to float array in cpu memory, throws if in gpu memory
    float* cpuData() const; 
    //gives pointer to float array in gpu memory, throws if in cpu memory
    float* gpuData() const;
    //moves data to gpu if not alraedy then returns the data pointer
    float* gpuDataForce(); 
    //moves data to cpu if not already then returns the data pointer
    float* cpuDataForce(); //

    //file interactions for saving and restoring tensors

    //extracts tensor information from a binary input, expects the current position to start at the tensors data, extracting from files must be precise
    void readBinary(std::ifstream& iF);
    //writes the tensors into binary assuming a binary output file, writes to wherever the output stream is so sequential tensor writes to the same open file can be read sequentially later
    void writeBinary(std::ofstream& oF);
    //clean write for readability, converts all 2 or greater dimensional tensors to 2d for the display
    void writeTensor(std::ofstream& oF);

    //public member variables
    int size, n; //total number of elements and the highest order dimension
    std::vector<int> dimensions; //dimension sizes
    float* data;




private:
    //enum to designate the CPU and GPU
    TensorLocation device;
    //pointer to the memory location of all the values
};

#endif