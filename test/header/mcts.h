#ifndef MCTS_H
#define MCTS_H

#include "../chess/header/chessState.h"
#include "tensor.h"
#include "tensorization.h"
#include "neuralNetwork.h"
#include "../chess/header/zobristKeys.h"
#include <fstream>
#include <unordered_map>


struct Edge{
  Edge(double prediction);
  int n; //times this edge has been selected
  double w, p; //total value w and prior probability p
};

struct Node{
  Node();
  Node(Node* p);
  Node* parent;
  std::map<uint16_t, std::unique_ptr<Edge>> children;
  int n; //times node is traversed
  bool expanded; //if this node has been expanded yet
};


class MCTS{
public:
  MCTS(NeuralNetwork& net, std::string out, const int thres);
  uint16_t search(chessState& init);
  uint16_t searchStore(chessState& init);
  double expand(Node& node, chessState& state, std::map<uint64_t, std::unique_ptr<Node>>& tree);
  double selectionRecurse(Node& node, chessState& state, std::map<uint64_t, std::unique_ptr<Node>>& tree);
  
private:
  NeuralNetwork& network;
  std::vector<std::string> fenLog;
  std::string oF;
  const int threshold;
  int moveConversion(chessState& state, uint16_t move);
};


#endif