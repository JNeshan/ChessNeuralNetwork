#ifndef GENERATOR_H
#define GENERATOR_H

#include <vector>
#include <fstream>
#include <string>
#include "cuda_runtime.h"
#include "curand.h"
#include "tensor.h"

struct cuRAND;

class Generator{
public:
  Generator();
  static void tGen(const Tensor& T);
  static void dGen(const int s, float* data);

private:
  static curandGenerator_t cGen;

};

#endif