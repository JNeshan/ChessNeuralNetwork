#include "../header/mcts.h"

int MCTS::moveConversion(chessState& state, uint16_t move){
  int nPos = move & 0x3f, pos = (move >> 6) & 0x3f, file = pos % 8, rank = pos / 8, nFile = nPos % 8, nRank = nPos / 8; 
  char piece = state.pieceAt(pos);
  piece = tolower(piece);
  int ind = pos * 73;
  int moveType = 0, rChange, fChange;
  int offset = 1;

  switch (piece)
  {
  case 'n': 
  {
    rChange = nRank - rank;
    fChange = nFile - file;
    if(rChange > 0){ //decrements rChange if its positive to avoid gap created by no possible 0
      rChange--;
    }
    fChange = (fChange > 0) ? 1 : 0; //any specific rank change always has two possible file changes, so fChange just uses direction to differentiate the indices
    rChange += 2; //shifts rChanges minimum to 0 for indicing
    moveType += (2 * rChange); //each possible rChange has 2 possible fChanges
    moveType += fChange;
    ind += moveType;
    break;
  }
  case 'q':
  case 'r':
  case 'b':
  case 'k':
  case 'p':
  {
    offset = 8; //8 indices created from knight moves
    int proPiece = (move >> 12) & 0x3f, promo = 0; //promotion checking
    if(proPiece){ //non-zero promotion flag implies pawn moves and promotes
      offset = 64; //64 planes before promotion moves
      if(proPiece & 0b0001){
        promo = 0;
      }
      else if(proPiece & 0b0010){
        promo = 1;
      }
      else if(proPiece & 0b0100){
        promo = 2;
      }
      else{ //one specific promotion is same as none, saves space
        offset = 8;
        promo = 0;
      }
    }
    rChange = nRank - rank;
    fChange = nFile - file;
    int direction = 0;
    //sets value for moves direction
    if(abs(rChange) == abs(fChange)){ //diagonals always have equal distances
      direction = (rChange > 0) ? direction = 2 : direction = 0;
      direction = (fChange > 0) ? direction += 1 : direction += 0;
    }
    else if(rChange){ 
      direction = (rChange > 0) ? direction = 4 : direction = 5;
    }
    else{
      direction = (fChange > 0) ? direction = 6 : direction = 7;
    }
    moveType = direction * 7 + std::max(abs(rChange), abs(fChange));
    moveType += (3 * promo);
    ind += moveType;
    ind += offset;
    break;
  }
  default:
  {
    throw("Bad pieceAt return");
  }
  }
  return ind;
}

Edge::Edge(double prediction) : w(prediction), n(0), p(w){}

Node::Node() : parent(this), expanded(false), n(0){}

Node::Node(Node* p) : parent(p), expanded(false), n(0){}

MCTS::MCTS(NeuralNetwork& net, std::string out, const int thres) : network(net), oF(out), threshold(thres){}

uint16_t MCTS::search(chessState& init){
  Tensorization TEN;
  Tensor T = TEN.tensorize(init);
  return 0ULL;
}

uint16_t MCTS::searchStore(chessState& init){
  std::ofstream output(oF);
  if(!output.is_open()){
    throw("Output file failed to open in MCTS");
  }
  std::map<uint64_t, std::unique_ptr<Node>> stateTree;
  uint64_t rtKey = init.getKey();
  uint64_t curKey = rtKey;
  stateTree[rtKey] = std::make_unique<Node>();
  Node* n = stateTree.at(rtKey).get();
  for(int i = 0; i < threshold; i++){
    chessState stateCpy = init;
    this->selectionRecurse(*n, stateCpy, stateTree);
  }
  int high = -1; 
  uint16_t move;
  for(auto& key : n->children){
    auto& edge = *key.second.get();
    if(edge.n > high){
      move = key.first;
      high = edge.n;
    }
  }
  return move;
}

double MCTS::selectionRecurse(Node& node, chessState& state, std::map<uint64_t, std::unique_ptr<Node>>& tree){
  if(node.expanded == false){
    return this->expand(node, state, tree);
  }
  //select move from nodes edges
  double high = -1;
  uint16_t move;
  const double balance = 1; //tweak
  Edge* traversed = nullptr;

  for(auto& key : node.children){
    auto& edge = *key.second.get();
    double q = 0;
    if(edge.n){
      q = edge.w / edge.n;
    }
    double u = balance * edge.p * (sqrt(node.n) / (1 + edge.n));
    double v = q + u;
    if(v > high){
      traversed = key.second.get();
      high = v;
      move = key.first;
    }
  }
  state.updateBoard(move);
  if(tree.find(state.getKey()) == tree.end()){
    tree[state.getKey()] = std::make_unique<Node>(&node);
  }
  double v = this->selectionRecurse(*tree.at(state.getKey()).get(), state, tree);
  node.n++;
  traversed->n++;
  traversed->w -= v;
  return -v;
}

double MCTS::expand(Node& node, chessState& state, std::map<uint64_t, std::unique_ptr<Node>>& tree){
  if(node.n != 0){
    throw("Expanded node expanding");
  }
  node.n = 1;
  std::vector<uint16_t> moveList = state.getAllMovesBit();
  Tensor stateTensor = Tensorization::tensorize(state);
  auto [Policy, Value] = network.evaluate(stateTensor, false);
  Policy.cpuSend(); Value.cpuSend();
  for(auto move : moveList){
    chessState sim = state;
    sim.updateBoard(move);
    uint64_t key = sim.getKey();
    if(tree.find(key) == tree.end()){
      tree[key] = std::make_unique<Node>(&node);
      int ind = moveConversion(state, move);
      node.children.at(move) = std::make_unique<Edge>(Policy.cpuData()[ind]);
    }
  }
  return Value.cpuData()[0];
}