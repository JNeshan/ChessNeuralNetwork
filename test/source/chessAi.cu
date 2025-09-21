#include "../header/chessAi.h"
#include <math.h>

DataCollection::DataCollection(size_t size) : capacity(size){}



void DataCollection::add(std::vector<Tensor>& i, std::vector<Tensor>& p, std::vector<Tensor>& v){
  std::lock_guard<std::mutex> lock(this->gDataMutex);
  for(int j = 0; j < i.size(); j++){
    this->data.push_back(std::make_tuple(std::move(i[j]), std::move(p[j]), std::move(v[j])));
  }
  while(this->data.size() > capacity){
    data.pop_front();
  }
}

std::tuple<Tensor, Tensor, Tensor> DataCollection::sample(){
  std::vector<Tensor*> inp, pol, val;
  Tensor inpT, polT, valT;
  int sampleSize = 512;
  inp.reserve(sampleSize); pol.reserve(sampleSize); val.reserve(sampleSize);
  std::lock_guard<std::mutex> lock(this->gDataMutex);
  std::srand(time(0));
  for(int i = 0; i < sampleSize; i++){
    int r = rand() % this->data.size();
    inp.push_back(&std::get<0>(this->data[r]));
    pol.push_back(&std::get<1>(this->data[r]));
    val.push_back(&std::get<2>(this->data[r]));
  }

  inpT.batchBuild(inp);
  polT.batchBuild(pol);
  valT.batchBuild(val);
  inpT.gpuSend();
  polT.gpuSend();
  valT.gpuSend();
  return std::make_tuple(std::move(inpT), std::move(polT), std::move(valT));
}

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

ChessAI::~ChessAI(){

}

void ChessAI::train(){
  DataCollection gameCollection(10000);
  NeuralNetwork* net = this->network.get();
  std::unique_ptr<NeuralNetwork> evalNetPtr(std::make_unique<NeuralNetwork>(*net));
  std::atomic<int> refresh = 0; //tracking number of tensors trained since evaluation network was updated
  int threshold, trainingStages, trainingSize, threads, updCnt;
  std::ifstream iF("fens.txt");
  std::thread evaluationThread, trainingThread, observerThread;
  std::vector<std::thread> generatorThreads;
  std::mutex lEvalNetMutex, lMainNetMutex;
  std::shared_mutex lGenMutex;
  std::atomic<int> active = 0;
  std::atomic<bool> dataReady = false, run = true, evalWaiting = false;

  std::cout<<"Start"<<std::endl;

  std::vector<std::string> fens;
  std::string line;
  while(std::getline(iF, line)){
    if(!line.empty()){
      fens.push_back(line);
    }
  }
  iF.close();
  
  threshold = 200;
  trainingStages = 1;
  trainingSize = 1;
  threads = 16;
  updCnt = 500;

  while(true){

    trainingThread = std::thread([net, &lMainNetMutex, &gameCollection, &run, &dataReady, &refresh]{
      std::srand(std::time(0));
      while(!dataReady && run){
        ThreadControl::cout(std::to_string(refresh));
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
      std::cout<<"Training thread started"<<std::endl;
      while(run){
        std::unique_lock<std::mutex> netLock(lMainNetMutex);
        std::tuple<Tensor, Tensor, Tensor> tmp = gameCollection.sample();
        auto& [inp, pol, val] = tmp;
        Tensor inpT, polT, valT;
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
        auto threadId = std::this_thread::get_id();
        int numericThreadId = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
        std::srand(numericThreadId);
        std::shared_lock<std::shared_mutex> genLock(lGenMutex);
        
        while(run){
          genLock.unlock();
          if(evalWaiting){
            std::unique_lock<std::mutex> lock(lEvalNetMutex); //blocks until evaluation thread finishes updating the network
          }
          genLock.lock();
          int r = std::rand() % 960;
          std::string randFen = fens[r];
          MCTS game(*evalNetPtr, threshold);
          game.runGame(randFen, evalWaiting, genLock, lEvalNetMutex);
          gameCollection.add(game.input, game.policy, game.value);
          refresh += game.input.size();
          if(refresh > 1000){
            dataReady = true;
          }
        }
        ThreadControl::cout(std::string("Generator thread terminating"));
      });
    }

  evaluationThread = std::thread([net, &evalNetPtr, &lEvalNetMutex, &lMainNetMutex, &gameCollection, &refresh, &active, updCnt, &run, &evalWaiting, &lGenMutex, &dataReady]{
    std::this_thread::sleep_for(std::chrono::seconds(5));
    while(run){
      if(refresh > updCnt && dataReady){
        std::unique_lock<std::mutex> evalLock(lEvalNetMutex);
        evalWaiting = true;
        while(!lGenMutex.try_lock()){
          auto* evalNetwork = evalNetPtr.get();
          evalNetwork->evaluationLoop();
        }

        std::unique_lock<std::shared_mutex> genLock(lGenMutex);
        std::unique_lock<std::mutex> netLock(lMainNetMutex);
        evalNetPtr.reset();
        evalNetPtr = std::make_unique<NeuralNetwork>(*net);
        refresh = 0;
      }
      
      auto* evalNetwork = evalNetPtr.get();
      evalNetwork->evaluationLoop();
    }
    ThreadControl::cout(std::string("Evaluation thread terminating"));
  });

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