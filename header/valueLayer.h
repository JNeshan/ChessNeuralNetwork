#ifndef VALUELAYER_H
#define VALUELAYER_H

#include "layer.h"

template<typename T>
class ValueLayer : Layer {
public:
  ValueLayer(size_t size);
  void setValue(size_t index, T value);
  T getValue(size_t index) const;
  size_t getSize() const;
private:
  T* values;
  size_t size;
};

#endif