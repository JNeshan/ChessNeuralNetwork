#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor {
public:
    Tensor(int p, int x, int y);
    Tensor(int id); //to get preexisting tensor potentially with id
    Tensor(Tensor* r);
    ~Tensor();
    void getVector(std::vector<float>& o);

private:
    int planes, xD, yD;
    std::vector<float> tensor;
    
};

#endif // TENSOR_H