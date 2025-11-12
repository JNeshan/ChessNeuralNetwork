#ifndef MCTS_H
#define MCTS_H

#include "../chess/header/chessState.h"
#include "tensor.cuh"
#include "tensorization.h"
#include "neuralNetwork.h"
#include "threadControl.h"
#include "../chess/header/zobristKeys.h"
#include <fstream>
#include <unordered_map>
#include <future>
#include <thread>
#include <chrono>
#include <atomic>
#include <list>
#include <mutex>
#include <shared_mutex>


//holds the pending evaluation request details

struct Node;

struct Edge{
  Edge(Node* p, int i, uint16_t m);
  Node* parent;
  Node* child;
  uint16_t move;
  int ind; //policy index
  int n; //times this edge has been selected
  double w, p; //total value w and prior probability p
};

struct Node{
  Node();
  Node(Node* p);
  Node* parent;
  std::map<uint16_t, std::unique_ptr<Edge>> children;
  int n; //times node is traversed
  bool explored; //if this node has been explored yet
};

struct RequestPackage{
  RequestPackage(std::list<std::pair<Node*, uint16_t>> pen);
  std::list<std::pair<Node*, uint16_t>> pendingNodes;
  std::future<Tensor<__half>> valueFuture;
  std::future<Tensor<__half>> policyFuture;
};


class MCTS{
public:
  MCTS(NeuralNetwork& net, const int thres);
  uint16_t search(chessState& init);
  double expand(Node& parent, chessState& state, uint16_t move, std::list<std::pair<Node*, uint16_t>> path, std::map<uint64_t, std::unique_ptr<Node>>& tree, std::list<std::unique_ptr<RequestPackage>>& pending);
  //double expand(Node& parent, chessState& state, Edge* parentEdge, std::list<std::pair<Node*, uint16_t>> path, std::map<uint64_t, std::unique_ptr<Node>>& tree, std::list<std::unique_ptr<RequestPackage>>& pending);
  
  double selectionRecurse(Node& node, chessState& state, std::list<std::pair<Node*, uint16_t>>& path, std::map<uint64_t, std::unique_ptr<Node>>& tree, std::list<std::unique_ptr<RequestPackage>>& pending);
  void acceptEvaluations(std::list<std::unique_ptr<RequestPackage>>& pending);
  void runGame(std::string fen, std::atomic<bool>& evalWait, std::shared_lock<std::shared_mutex>& genLock, std::mutex& evalNetMutex);
  static std::atomic<int> savedStates;
  std::vector<Tensor<__half>> input, policy, value;

private:
  NeuralNetwork* network;
  std::vector<std::string> fenLog;
  const int threshold;
  int moveConversion(chessState& state, uint16_t move);
};


#endif