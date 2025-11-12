#include "../header/chessAi.h"
#include <math.h>

DataCollection::DataCollection(size_t size) : capacity(size){}

void DataCollection::add(std::vector<Tensor>& i, std::vector<Tensor>& p, std::vector<Tensor>& v){
  std::lock_guard<std::mutex> lock(this->gDataMutex); //locks mutex before modifying member
  for(int j = 0; j < i.size(); j++){
    this->data.push_back(std::make_tuple(std::move(i[j]), std::move(p[j]), std::move(v[j]))); 
  }
  while(this->data.size() > capacity){
    data.pop_front(); //maintains maximum capacity
  }
}

std::tuple<Tensor, Tensor, Tensor> DataCollection::sample(){
  std::vector<Tensor*> inp, pol, val;
  Tensor inpT, polT, valT; //blank tensors to convert sets of tensors into one large batched tensor
  int sampleSize = 512; //number of training tensors retrieved for processing
  inp.reserve(sampleSize); pol.reserve(sampleSize); val.reserve(sampleSize);
  std::lock_guard<std::mutex> lock(this->gDataMutex); //lock mutex to prevent race condition
  std::srand(time(0)); 
  for(int i = 0; i < sampleSize; i++){
    int r = rand() % this->data.size(); //finds a random index and takes it
    inp.push_back(&std::get<0>(this->data[r]));
    pol.push_back(&std::get<1>(this->data[r]));
    val.push_back(&std::get<2>(this->data[r]));
  }

  inpT.batchBuild(inp); //uses the set to construct one large batched tensor
  polT.batchBuild(pol);
  valT.batchBuild(val);
  inpT.gpuSend(); //ensuring gpu stored before returning
  polT.gpuSend();
  valT.gpuSend();
  return std::make_tuple(std::move(inpT), std::move(polT), std::move(valT));
}
//constructs the neural network structure
ChessAI::ChessAI(){
  std::vector<std::unique_ptr<Layer>> tmpBody, tmpValue, tmpPolicy;
  
  tmpBody.push_back(std::make_unique<ConvolutionLayer>(ConvolutionLayer(256, 17, 3, 3, 1)));
  tmpBody.push_back(std::make_unique<NormalizationLayer>(NormalizationLayer(true, 256)));
  tmpBody.push_back(std::make_unique<ReLULayer>());

  for(int i = 0; i < 20; i++){
    tmpBody.push_back(std::make_unique<ResidualBlock>());
  }

  tmpPolicy.push_back(std::make_unique<ConvolutionLayer>(ConvolutionLayer(2, 256, 1, 1, 0)));
  tmpPolicy.push_back(std::make_unique<NormalizationLayer>(NormalizationLayer(true, 2)));
  tmpPolicy.push_back(std::make_unique<ReLULayer>());
  tmpPolicy.push_back(std::make_unique<FlattenLayer>());
  tmpPolicy.push_back(std::make_unique<DenseLayer>(DenseLayer(128, 4672)));
  tmpPolicy.push_back(std::make_unique<SoftmaxLayer>()); 

  tmpValue.push_back(std::make_unique<ConvolutionLayer>(ConvolutionLayer(1, 256, 1, 1, 0)));
  tmpValue.push_back(std::make_unique<NormalizationLayer>(NormalizationLayer(true, 1)));
  tmpValue.push_back(std::make_unique<ReLULayer>());
  tmpValue.push_back(std::make_unique<FlattenLayer>());
  tmpValue.push_back(std::make_unique<DenseLayer>(DenseLayer(64, 256)));
  tmpValue.push_back(std::make_unique<ReLULayer>());
  tmpValue.push_back(std::make_unique<DenseLayer>(DenseLayer(256, 1)));
  tmpValue.push_back(std::make_unique<tanhLayer>());
  this->network = std::make_unique<NeuralNetwork>(tmpBody, tmpPolicy, tmpValue);
}
//smart pointers sure are handy
ChessAI::~ChessAI(){

}
//initiates self teaching loop
void ChessAI::train(){
  DataCollection gameCollection(50000); //represents maximum stored sample size
  NeuralNetwork* net = this->network.get(); //raw ptr
  std::unique_ptr<NeuralNetwork> evalNetPtr(std::make_unique<NeuralNetwork>(*net)); //creates a unique copy of the network to use strictly for evaluations
  std::atomic<int> refresh = 0; //tracking number of tensors trained since evaluation network was updated
  std::ifstream iF("fens.txt");
  //evaluation thread used to run quick evaluations of games
  //training thread used to perform training runs of the network
  //observer thread used for debugging
  //generator threads used for playing games to use for creating training data
  std::thread evaluationThread, trainingThread, observerThread; 
  std::vector<std::thread> generatorThreads;
  //evalnet mutex prevents race condtions for the evaluation network copy
  //mainnet mutex prevents race conditions for the primary member network
  std::mutex lEvalNetMutex, lMainNetMutex;
  std::shared_mutex lGenMutex;
  //
  std::atomic<int> active = 0;
  //dataReady marks if enough training data has generated to begin the training loop
  //run controls whether all the threads loops keep running
  //evalWaiting is used to signal the generator threads to block when the evaluation network needs to update but still allowing the evaluation to run,
  //which is necessary for the generator threads to reach their mutex
  std::atomic<bool> dataReady = false, run = true, evalWaiting = false;

  std::cout<<"Start"<<std::endl;
  //creates set of fens used to generate initial board states for the games that are played
  std::vector<std::string> fens;
  std::string line;
  while(std::getline(iF, line)){
    if(!line.empty()){
      fens.push_back(line);
    }
  }
  iF.close();
  
  int threshold = 300; //search depth of mcts
  int trainingStages = 1; //unused
  int trainingSize = 1; //unused
  int threads = 30; //number of generator threads
  int updCnt = 2000; //number of states trained on before the evaluation network should be updated
  //change updCnt and refresh to reflect number of training loops done rather than number of game states

  while(true){
    
    trainingThread = std::thread([net, &lMainNetMutex, &gameCollection, &run, &dataReady, &refresh]{
      std::srand(std::time(0));
      while(!dataReady && run){ //training thread idles until starting threshold is met
        //ThreadControl::cout(std::to_string(refresh));
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
      std::cout<<"Training thread started"<<std::endl;
      while(run){
        std::unique_lock<std::mutex> netLock(lMainNetMutex); //enforces lock before training
        std::tuple<Tensor, Tensor, Tensor> tmp = gameCollection.sample(); //generates sample collection
        auto& [inp, pol, val] = tmp; //
        //Tensor inpT, polT, valT;
        net->train(inp, pol, val, 0.2);
      }
      ThreadControl::cout(std::string("Training thread terminating"));
    });

    //observerThread = std::thread([&run]{
    //  std::string command = ThreadControl::cin();
    //  if(command == "quit" || command == "q"){
    //    run = false;
    //  }
    //  else{
    //    ThreadControl::cout(std::string("Invalid command"));
    //  }
    //  ThreadControl::cout(std::string("Observer thread terminating"));
    //});
    
    for(int i = 0; i < threads; i++){
      generatorThreads.emplace_back([&evalNetPtr, &lEvalNetMutex, &gameCollection, &active, &run, &evalWaiting, &lGenMutex, &fens, &refresh, threshold, &dataReady]{
        auto threadId = std::this_thread::get_id(); //threads unique id, used for generating game output file names for testing
        int numericThreadId = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
        std::srand(numericThreadId); //its random seeded with random so why not
        //creates a shared lock on the generator mutex so all the generator threads hold it
        //this enforces every generator thread needing to be done before another thread can place a strict mutex
        std::shared_lock<std::shared_mutex> genLock(lGenMutex); 
        
        while(run){
          //unlocks so the only time it isn't locking it is after its finished a game and before it starts a new one
          //this is so the evaluation network is only updated after every generator thread has completed the game they're playing
          //which is necessary to ensure each game uses the same version of the network the whole way through
          genLock.unlock();
          if(evalWaiting){ //if evaluation thread is waiting for the generator threads to terminate
            std::unique_lock<std::mutex> lock(lEvalNetMutex); //blocks each generator until evaluation thread finishes updating the network then immediately unblocks
          }
          genLock.lock();
          int r = std::rand() % fens.size(); //grabs random starting fen
          std::string randFen = fens[r];
          MCTS game(*evalNetPtr, threshold); //game simulation object creation
          game.runGame(randFen, evalWaiting, genLock, lEvalNetMutex); 
          gameCollection.add(game.input, game.policy, game.value);
          refresh += game.input.size();
          if(refresh > 512){ //checks if enough training data has been generated to tell the training thread to start
            dataReady = true;
          }
        }
        ThreadControl::cout(std::string("Generator thread terminating"));
      });
    }

  evaluationThread = std::thread([net, &evalNetPtr, &lEvalNetMutex, &lMainNetMutex, &gameCollection, &refresh, &active, updCnt, &run, &evalWaiting, &lGenMutex, &dataReady]{
    std::this_thread::sleep_for(std::chrono::seconds(5)); //initial buffer
    while(run){
      //condition currently updates based on number of samples collected and not amount of training performed, fix
      if(refresh > updCnt && dataReady){
        //locks evalnet mutex so the generator threads block, must always be free when evaluationThread is not blocking it
        //must be locked before evalWaiting is updated
        std::unique_lock<std::mutex> evalLock(lEvalNetMutex); 
        evalWaiting = true;
        //checks if gen mutex can be locked without actually locking, loop continues while this is false or until the mutex can be locked
        //standard evaluation loop runs until all generator games are finished since it is needed for them to do so
        while(!lGenMutex.try_lock()){
          auto* evalNetwork = evalNetPtr.get();
          evalNetwork->evaluationLoop();
        }
        //creates a unique lock over the generator mutex, probably redundant with the other mutexs
        std::unique_lock<std::shared_mutex> genLock(lGenMutex);
        //locks mainnet mutex to avoid race condition with training thread
        std::unique_lock<std::mutex> netLock(lMainNetMutex);
        //deletes the old copy
        evalNetPtr.reset();
        //copies the most current network state
        evalNetPtr = std::make_unique<NeuralNetwork>(*net);
        refresh = 0; //resets refresh threshold
        evalWaiting = false; //toggling atomic
      }
      //constantly runs evaluation network passes
      auto* evalNetwork = evalNetPtr.get(); 
      evalNetwork->evaluationLoop();
    }
    ThreadControl::cout(std::string("Evaluation thread terminating"));
  });
    //waits for all threads to finish before terminating
    while(evaluationThread.joinable()){
      evaluationThread.join();
    }
    while(observerThread.joinable()){
      observerThread.join();
    }
    while(trainingThread.joinable()){
      trainingThread.join();
    }
    for(auto& thread : generatorThreads){
      while(thread.joinable()){
        thread.join();
      }
    }
    return;
  }
}

void ChessAI::generateValues(){
  std::cout<<"Generating"<<std::endl;
  auto* netPtr = this->network.get();
  
  for(auto& ptr : netPtr->body){
    auto* layer = ptr.get();
    layer->genTensorData();
  }
  for(auto& ptr : netPtr->value){
    auto* layer = ptr.get();
    layer->genTensorData();
  }
  for(auto& ptr : netPtr->policy){
    auto* layer = ptr.get();
    layer->genTensorData();
  }
}

void ChessAI::loadLayers(std::ifstream& iF){
  std::ofstream oF("1CleanOut.txt");
  for(auto& l : this->body){
    l->loadTensor(iF);
    
    //l->cleanSave(oF);
  }
  for(auto& l : this->value){
    l->loadTensor(iF);
    
    //l->cleanSave(oF);
  }
  for(auto& l : this->policy){
    l->loadTensor(iF);
    
    //l->cleanSave(oF);
  }
  oF.close();
}

void ChessAI::saveLayers(std::ofstream& oF){
  std::ofstream outF("cleanOut.txt");
  for(auto& l : this->body){
    l->saveTensor(oF);
    //l->cleanSave(outF);
    
  }
  for(auto& l : this->value){
    l->saveTensor(oF);
    //l->cleanSave(outF);
  }
  for(auto& l : this->policy){
    l->saveTensor(oF);
    //l->cleanSave(outF);
  }
}