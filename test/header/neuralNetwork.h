//neuralNetwork.h
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "tensor.h"
#include "layer.h"
#include "optimizer.h"

class NeuralNetwork{
public:
  NeuralNetwork(std::vector<Layer*>& b, std::vector<Layer*>& pH, std::vector<Layer*>& vH);
  std::pair<Tensor, Tensor> evaluate(const Tensor& inp, bool train = false);
  void train(const Tensor& inp, const Tensor& correctValue, const Tensor& correctPolicy, const int lR);
  void loadLayers(std::ofstream& oF); //loads layers tensors from file
  void saveLayers(std::ofstream& oF); //saves layers tensors to file
  void backPropagate(const Tensor& v, const Tensor& p);
  
private:
  Optimizer optimize;
  std::vector<Layer*> body;
  std::vector<Layer*> policy;
  std::vector<Layer*> value;
};



#endif