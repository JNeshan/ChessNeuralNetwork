#ifndef CHESSAI_H
#define CHESSAI_H
#include <iostream>

#include <fstream>
#include "tensor.h"
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
  void add(std::vector<Tensor>& i, std::vector<Tensor>& p, std::vector<Tensor>& v);
  std::tuple<Tensor, Tensor, Tensor> sample();
  size_t capacity;
  std::mutex gDataMutex;
  std::deque<std::tuple<Tensor, Tensor, Tensor>> data;
};

class ChessAI{
public:
  ChessAI();
  ~ChessAI();
  void play();
  void train();
  void generateValues();
  void loadLayers(std::ifstream& iF);
  void saveLayers(std::ofstream& oF);
private:
  std::vector<Layer*> body, policy, value;
  std::unique_ptr<NeuralNetwork> network;
  
};

#endif