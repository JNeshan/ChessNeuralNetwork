#ifndef CHESSAI_H
#define CHESSAI_H
#include <iostream>

#include <fstream>
#include "tensor.cuh"
#include "matriceMath.h"
#include "generator.h"
#include "optimizer.h"
#include "layer.h"
#include "denseLayer.h"
#include "reluLayer.h"
#include "convolutionLayer.h"
#include "tanhLayer.h"
#include "softmaxLayer.h"
#include "normalizationLayer.h"
#include "residualBlock.h"
#include "flattenLayer.h"
#include "mcts.h"
#include <thread>
#include <future>
#include <shared_mutex>

struct DataCollection{
  DataCollection(size_t size);
  //adds training data tensor<float>s to the global collection
  void add(std::vector<Tensor<float>>& i, std::vector<Tensor<float>>& p, std::vector<Tensor<float>>& v);
  //grabs a random selection of the stored training data
  std::tuple<Tensor<float>, Tensor<float>, Tensor<float>> sample();
  size_t capacity;
  //used to prevent race conditions for asynchronous reading and writing processes
  std::mutex gDataMutex;
  //global training data collection structure
  std::deque<std::tuple<Tensor<float>, Tensor<float>, Tensor<float>>> data;
};

class ChessAI{
public:
  ChessAI();
  ~ChessAI();
  void play();
  //starts and controls the self generation and training loop
  void train();
  //populates the layers with random initial values
  void generateValues();
  void loadLayers(std::ifstream& iF);
  void saveLayers(std::ofstream& oF);
private:
  std::vector<Layer*> body, policy, value;
  std::unique_ptr<NeuralNetwork> network;
};

#endif