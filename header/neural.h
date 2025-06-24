#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include <cstdint>
#include <ctime>
#include "convolutionLayer.h"
#include "denseLayer.h"
#include "reluLayer.h"

enum class LayerType {CON, RELU, DENSE, OUTX, OUTY};


class NeuralNetwork {
public:
  NeuralNetwork(std::string fName);
  ~NeuralNetwork();
  void RunNetwork(Tensor& input); //to start a run of the network,temporary placeholder
  void SaveNetwork();
  void SetLayers(const std::vector<std::unique_ptr<Layer>>& l, const std::vector<LayerType>& lT);
  void initialize();
  void encode();


private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<LayerType> layTypes;
  std::string file; //file name used for storing tensors
  std::ifstream rF;
  std::ofstream wF;
  void saveTensors();
  void loadTensors();
  bool ready;
  
};

#endif // NEURAL_H