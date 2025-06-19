#ifndef INITMATRIX_H
#define INITMATRIX_H

class InitMatrix {
public:
    InitMatrix();
    ~InitMatrix();

private:
    int rows;
    int cols;
    int planes;
    void allocateMatrix();
    void deallocateMatrix();
};

#endif // INITMATRIX_H