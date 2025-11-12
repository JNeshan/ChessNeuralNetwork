//neuralNetwork.h
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <future>
#include "tensor.cuh"
#include "layer.h"
#include "optimizer.h"
#include "tensorization.h"
#include "../chess/header/chessState.h"
#include "threadControl.h"
#include <queue>
#include <mutex>
#include <thread>


struct Request{
  Request(Tensor<__half>& inp);
  Request(chessState& state);
  Tensor<__half> inpState;
  std::promise<std::pair<Tensor<__half>, Tensor<__half>>> promise;
  std::promise<Tensor<__half>> pPromise, vPromise;
  std::future<Tensor<__half>> getPolicyFuture() {return pPromise.get_future();}
  std::future<Tensor<__half>> getValueFuture() {return vPromise.get_future();}
  operator const Tensor<__half>&() const {return inpState;}
  operator Tensor<__half>&() {return inpState;}
};



class NeuralNetwork{
public:
  NeuralNetwork(std::vector<std::unique_ptr<Layer>>& b, std::vector<std::unique_ptr<Layer>>& pH, std::vector<std::unique_ptr<Layer>>& vH);
  NeuralNetwork(const NeuralNetwork& r);
  ~NeuralNetwork();
  std::pair<Tensor<__half>, Tensor<__half>> evaluate(const Tensor<__half>& inp, bool train = false);
  void batchEvaluate(std::vector<std::unique_ptr<Request>>& r, Tensor<__half>& inp, bool train = false);
  void train(const Tensor<__half>& inp, const Tensor<__half>& correctValue, const Tensor<__half>& correctPolicy, const int lR);
  void loadLayers(std::ofstream& oF); //loads layers tensors from file
  void saveLayers(std::ofstream& oF); //saves layers tensors to file
  void backPropagate(Tensor<__half>& v, Tensor<__half>& p);
  void evaluationLoop();
  void evaluationRequest(std::unique_ptr<Request>& req);
  void shutDown();
  std::vector<std::unique_ptr<Layer>> body;
  std::vector<std::unique_ptr<Layer>> policy;
  std::vector<std::unique_ptr<Layer>> value;
  friend class Layer;

private:
  Optimizer optimize;
  
  std::mutex gQueueMutex; //mutex for the queue 
  const int batchSize;
  bool run;
  std::queue<std::unique_ptr<Request>> queue;
};



#endif