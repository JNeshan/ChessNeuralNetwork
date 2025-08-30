#ifndef TENSORIZATION_H
#define TENSORIZATION_H

#include "../chess/header/chessState.h"
#include "tensor.h"

class Tensorization{
public:
  Tensorization();
  static Tensor tensorize(const chessState& state);

private:

};

#endif