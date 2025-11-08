#ifndef TENSORIZATION_H
#define TENSORIZATION_H

#include "../chess/header/chessState.h"
#include "tensor.cuh"
class Tensorization{
public:
  Tensorization();
  static Tensor<float> tensorize(const chessState& state);

private:

};

#endif