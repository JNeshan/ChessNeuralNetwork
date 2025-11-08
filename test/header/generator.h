#ifndef GENERATOR_H
#define GENERATOR_H

#include <vector>
#include <fstream>
#include <string>
#include "cuda_runtime.h"
#include "curand.h"
#include "tensor.cuh"

struct cuRAND;

class Generator{
public:
  Generator();
  static void tGen(Tensor<float>& T);
  //populates data with random values
  static void dGen(const int s, float* data);
  //populates data with ascending values starting at 0
  static void aGen(const int s, float* data);
  //populates data with a specified value
  static void vGen(const int s, const int v, float* data);
  //copies data values
  static void copy(const int s, const float* r, float* data);

private:
  thread_local static curandGenerator_t cGen;

};

#endif