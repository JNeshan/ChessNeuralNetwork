
#include "../header/mcts.h"
#include <chrono>

std::atomic<int> MCTS::savedStates{0};

int MCTS::moveConversion(chessState& state, uint16_t move){
  int nPos = move & 0x3f, pos = (move >> 6) & 0x3f, file = pos % 8, rank = pos / 8, nFile = nPos % 8, nRank = nPos / 8; //constants for the different number of possibilities that determine the index
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
    throw std::runtime_error("Bad pieceAt return");
  }
  }
  if(ind < 0 || ind > 4671){
    throw std::runtime_error("Bad move conversion");
  }
  return ind;
}

RequestPackage::RequestPackage(std::list<std::pair<Node*, uint16_t>> pen) : pendingNodes(pen){
}

Edge::Edge(Node* p, int i, uint16_t m): w(0), n(0), p(0), ind(i), move(m), parent(p), child(nullptr){
  
}

Node::Node() : parent(this), explored(false), n(0){}

Node::Node(Node* p) : parent(p), explored(false), n(0){}

MCTS::MCTS(NeuralNetwork& net, const int thres) : network(&net), threshold(thres){}

uint16_t MCTS::search(chessState& init){  
  std::map<uint64_t, std::unique_ptr<Node>> stateTree; //stores all the board state nodes
  uint64_t rtKey = init.getKey(); //zobrist hash key for the initial board state
  stateTree[rtKey] = std::make_unique<Node>(); //initial node
  Node* node = stateTree[rtKey].get(); //reference
  std::list<std::unique_ptr<RequestPackage>> pending; //holds the request details to check and retrieve when network evaluations are done
  this->expand(*node, init, 0, std::list<std::pair<Node*, uint16_t>>({{node, 0}}), stateTree, pending); //expanding initial
  for(int i = 0; i < threshold; i++){
    this->acceptEvaluations(pending);
    std::list<std::pair<Node*, uint16_t>> path = {};
    chessState stateCpy = init; //deep copies initial state
    this->selectionRecurse(*node, stateCpy, path, stateTree, pending);
  }
  //after threshold reached waits until every state recieves its evaluation before finishing the algorithm
  while(!pending.empty()){
    RequestPackage* pendPtr = pending.front().get();
    pendPtr->policyFuture.wait();
    this->acceptEvaluations(pending);
  }
  //cache high score
  int high = -1; 
  //cache best move
  uint16_t move;
  this->policy.push_back(Tensor({1, 4672}, TensorLocation::CPU)); //creating correct policy tensor
  float* data = this->policy[this->policy.size()-1].cpuData(); 
  for(auto& key : node->children){
    auto& edge = *key.second.get();
    int ind = this->moveConversion(init, key.first);
    float visitRatio = 0;
    if(edge.n){
      visitRatio = float(node->n) / float(edge.n);
    }
    data[ind] = visitRatio;
    if(edge.n > high){
      move = key.first;
      high = edge.n;
    }
  }
  return move;
}

double MCTS::selectionRecurse(Node& node, chessState& state, std::list<std::pair<Node*, uint16_t>>& path, std::map<uint64_t, std::unique_ptr<Node>>& tree, std::list<std::unique_ptr<RequestPackage>>& pending){
  double eval = state.winner(); //checks if state is terminal
  if(abs(eval) == 1.0){ //win or loss
    return -eval;
  }
  else if(eval == 0.5){ //draw
    return 0;
  }

  node.explored = true; //marks the passed node as explored

  //select move from nodes edges
  double high = -9999; //tracks the highest score
  uint16_t move; //cache best move
  const double balance = 1; //tweakable, to balance the q and u values
  Edge* traversed = nullptr; //pointer to best edge yet

  for(auto& [potMove, edgePtr] : node.children){ //iterates over the move, edge map
    if(edgePtr == nullptr){ //throws if node has no children
      throw std::runtime_error("Edge is nullptr");
    }
    auto& edgeRef = *edgePtr.get(); //creates reference to edge
    double q = 0; //policy score
    if(edgeRef.child != nullptr){ //if edge has a child set
      if(edgeRef.child->explored){ //avoids if already explored
        continue;
      }
    }
    else{
      chessState tmp = state; //copies the state for simulation
      tmp.updateBoard(potMove); //simulates move
      uint64_t key = tmp.getKey();
      if(tree.find(key) != tree.end()){ //checks if child reference needs to be added
        edgeRef.child = tree[key].get(); //adds reference
        if(edgeRef.child->explored){ //skips if explored
          continue;
        }
      }
    }

    if(edgeRef.n){ //no policy score if edge never traversed 
      q = edgeRef.w / edgeRef.n;
    }
    //calculating node score
    double u = balance * edgeRef.p * (sqrt(node.n) / (1 + edgeRef.n)); 
    double v = q + u;
    if(v > high){ //if score is better than current best, updates
      traversed = &edgeRef; 
      high = v;
      move = potMove;
    }
  }

  if(traversed == nullptr){ //indicates no moves leading to untraversed states, every possible move creates a loop
    node.explored = false;
    node.n++;
    return 0; //treat as draw score
  }

  if(move == 0){
    //ThreadControl::cout(std::string("0 move"));
  }
  path.push_back({&node, move}); //adds selected move to path
  state.updateBoard(move); //updates running simulation state
  double v2; //score variable
  if(tree.find(state.getKey()) == tree.end()){
    v2 = this->expand(node, state, move, path, tree, pending);
  }
  else{
    Node* child = tree[state.getKey()].get();
    v2 = this->selectionRecurse(*child, state, path, tree, pending);
  }

  node.explored = false; //sets as unexplored during backpropagation
  node.n++;
  traversed->n++;
  traversed->w += v2;
  return -v2;
}

double MCTS::expand(Node& parent, chessState& state, uint16_t move, std::list<std::pair<Node*, uint16_t>> path, std::map<uint64_t, std::unique_ptr<Node>>& tree, std::list<std::unique_ptr<RequestPackage>>& pending){
  Node* node;
  if(!move){
    node = &parent;
  }
  else{
    tree[state.getKey()] = std::make_unique<Node>(&parent);
    node = tree[state.getKey()].get();
    auto* edge = parent.children[move].get();
    edge->child = node;
    path.push_back({node, 0});
  }
  
  std::vector<uint16_t> moveList = state.getAllMovesBit();
  Tensor inp = Tensorization::tensorize(state); //tensorizing the game state for the network to use
  std::unique_ptr<Request> reqPtr = std::make_unique<Request>(state); //creating a request object in smart pointer
  Request* req = reqPtr.get(); //refernece pointer
  pending.push_back(std::make_unique<RequestPackage>(path)); //adding new RequestPackage to queue 
 
  RequestPackage* pack = pending.back().get();
  pack->policyFuture = req->getPolicyFuture();
  pack->valueFuture = req->getValueFuture();
  this->network->evaluationRequest(reqPtr); //sends its tensor and promises to be evaluated in a larger batch, swaps the passed ptr 
  reqPtr = nullptr; //emptied

  for(auto mv : moveList){
    int ind = this->moveConversion(state, mv);
    node->children[mv] = std::make_unique<Edge>(node, ind, mv);
  }
  path.pop_back(); //popping off the destination node, it didn't traverse an edge
  while(!path.empty()){
    node = path.back().first;
    uint16_t mv = path.back().second;
    path.pop_back();
    auto& edgeRef = node->children[mv];
    edgeRef->w -= 1.1;
  }
  return 0; //loss applied manually so no other value is applied yet
}

void MCTS::acceptEvaluations(std::list<std::unique_ptr<RequestPackage>>& pending){
  while(!pending.empty()){
    RequestPackage* result = pending.front().get();
    if(result->policyFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready){
      return;
    }
    Tensor P = result->policyFuture.get();
    Tensor V = result->valueFuture.get();
    P.cpuSend(); V.cpuSend(); //sending to cpu memory incase it isnt
    if(P.size != 4672 || V.size != 1){
      throw std::runtime_error("RequestPackage gave bad tensors");
    }
    if(result->pendingNodes.size() == 0){
      throw std::runtime_error("Pending nodes not populated");
    }
    Node* node = result->pendingNodes.back().first;

    if(node == nullptr){
      throw std::runtime_error("RequestPackage contained null ptr");
    }

    for(auto& [move, edgePtr] : node->children){ //iterates over the evaluated states edges
      Edge* edge = edgePtr.get(); 
      edge->p = P.cpuData()[edge->ind]; //gives each edge their respective policy value
    }

    int v = -V.cpuData()[0];

    result->pendingNodes.pop_back(); //removes the final state node since didn't use outgoing edge
    while(!result->pendingNodes.empty()){ 
      node = result->pendingNodes.back().first;
      uint16_t move = result->pendingNodes.back().second;
      result->pendingNodes.pop_back();
      auto* edge = node->children[move].get();
      edge->w += (1.1 + v); //removes the decentivization and adds the correct score
      v = -v; //flips the win value
    }

    result = nullptr;
    pending.pop_front(); //removes the completed package
  }
}

void MCTS::runGame(std::string fen, std::atomic<bool>& evalWait, std::shared_lock<std::shared_mutex>& genLock, std::mutex& evalNetMutex){
  chessState game(fen);
  int moves = 0;
  double eval = 0;
  auto threadId = std::this_thread::get_id();
  int numericThreadId = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
  std::string threadIdStr = std::to_string(numericThreadId);
  std::string output = std::string("ChessGames/game") + threadIdStr + std::string(".txt");
  std::ofstream oF(output);
  oF <<"Start\n"<<fen<<std::endl;
  while(!eval){
    if(evalWait){
      genLock.unlock();
      std::unique_lock<std::mutex> lock(evalNetMutex);
      genLock.lock();
    }
    moves++;
    uint16_t move = this->search(game);
    this->input.push_back(Tensorization::tensorize(game));
    std::string moveString = game.readMove(move);
    oF<<moveString<<" "<<std::endl;
    game.updateBoard(move);
    eval = game.winner();
  }
  oF<<"Game End"<<std::endl;
  //ThreadControl::cout(std::string("\nGame ended"));
  oF.close();
  if(eval == 0.5){
    Tensor V({1, 1}, TensorLocation::CPU); //0
    this->value = std::vector<Tensor>(this->input.size(), V);
  }
  else{
    std::vector<Tensor> alt({Tensor({1, 1}, TensorLocation::CPU), Tensor({1, 1}, TensorLocation::CPU)});
    alt[0].cpuData()[0] = 1; alt[1].cpuData()[0] = -1;
    bool flip = false;
    int end = value.size();
    this->value.resize(this->input.size());
    for(int i = input.size()-1; i >= end; i--){
      this->value[i] = (alt[flip]);
      flip = !flip;
    }
  }
  //ThreadControl::cout(std::string("Tensors stored"));
}


