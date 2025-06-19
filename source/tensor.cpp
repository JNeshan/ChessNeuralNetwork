#include <stdexcept>
#include <vector>

Tensor::Tensor(int p, int x, int y) : planes(p), xD(x), yD(y) {
    if (p <= 0 || x <= 0 || y <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive integers.");
    }
    tensor.resize(p * x * y, 0.0f);
}

Tensor::Tensor(int id) {
    planes = 0;
    xD = 0;
    yD = 0;
    tensor.clear();
}

Tensor::Tensor(Tensor* r) {
    if (r == nullptr) {
        throw std::invalid_argument("Cannot copy from a null tensor.");
    }
    planes = r->planes;
    xD = r->xD;
    yD = r->yD;
    tensor = r->tensor;
}

Tensor::~Tensor() {}

void Tensor::getVector(std::vector<float>& o) {
    o = tensor;
}
