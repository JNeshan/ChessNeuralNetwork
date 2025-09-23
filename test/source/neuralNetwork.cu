#include "../header/neuralNetwork.h"
__global__ void ValueLossKernel(const float* eval, const float* correct, float* out, const int size){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size){
    out[idx] = 2 * (eval[idx] - correct[idx]);
  }
}

__global__ void PolicyLossKernel(const float* eval, const float* correct, float* out, const int size){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size){
    out[idx] = eval[idx] - correct[idx];
  }
}

Request::Request(Tensor& inp) : inpState(inp){
}

Request::Request(chessState& state){
  this->inpState = Tensor(Tensorization::tensorize(state));
}


NeuralNetwork::NeuralNetwork(std::vector<std::unique_ptr<Layer>>& b, std::vector<std::unique_ptr<Layer>>& pH, std::vector<std::unique_ptr<Layer>>& vH) : optimize(1), batchSize(4096){
  int i = 0;
  
  for(auto& ptr : b){
    this->body.push_back(nullptr);
    this->body[i].swap(ptr);
    i++;
    }
  
  i = 0;
  for(auto& ptr : pH){
    this->policy.push_back(nullptr);
    this->policy[i].swap(ptr);
    i++;
  }
  i = 0;
  for(auto& ptr : vH){
    this->value.push_back(nullptr);
    this->value[i].swap(ptr);
    i++;
  }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& r) : optimize(r.optimize), batchSize(r.batchSize){
  for(auto& ptr : r.body){
    auto* l = ptr.get();
    this->body.push_back(l->clone());
  }
  for(auto& ptr : r.value){
    auto* l = ptr.get();
    this->value.push_back(l->clone());
  }
  for(auto& ptr : r.policy){
    auto* l = ptr.get();
    this->policy.push_back(l->clone());
  }
}

NeuralNetwork::~NeuralNetwork(){
  if(this->run == true){
    std::runtime_error("Attempting to delete network while still running");
  }
  while(!this->queue.empty()){
    this->queue.pop();
  }
}

std::pair<Tensor, Tensor> NeuralNetwork::evaluate(const Tensor& inp, bool train){
  Tensor T = inp; 
  T.gpuSend();
  for(auto& layer : body){
    T = layer->forward(T, train);
  }

  Tensor P = std::move(T);
  Tensor V = P;

  for(int i = 0; i < policy.size(); i++){
    P = policy[i]->forward(P, train);
  }
  for(int i = 0; i < value.size(); i++){
    V = value[i]->forward(V, train);
  }
  //std::cout<<V.n<<" "<<P.n<<std::endl;
  return std::make_pair(V, P);
}

void NeuralNetwork::backPropagate(Tensor& v, Tensor& p){
  std::cout<<"Backpropagation started"<<std::endl;
  Tensor V = value[value.size()-1]->backward(v);
  Tensor P = policy[policy.size()-1]->backward(p);
  for(int i = value.size()-2; i >= 0; i--){
    V = value[i]->backward(V);
  }
  //std::cout<<"Value done"<<std::endl;
  for(int i = policy.size()-2; i >= 0; i--){
    P = policy[i]->backward(P);
  }
  if(V.size != P.size){
    throw std::runtime_error("Value and policy backpropagation outputs wrong");
  }
  //std::cout<<"Policy Done"<<std::endl;
  V.gpuAdd(P);
  
  for(int i = body.size()-1; i >= 0; i--){
    auto start = std::chrono::steady_clock::now();
    //std::cout<<i<<std::endl;
    V = body[i]->backward(V);
    auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout<<std::string("Time in body layer back: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  }

  
  //std::cout<<"Backpropagation ended"<<std::endl;
}

void NeuralNetwork::train(const Tensor& inp, const Tensor& correctValue, const Tensor& correctPolicy, const int lR){
  auto start = std::chrono::steady_clock::now();
    
  auto [evalValue, evalPolicy] = this->evaluate(inp, true); //runs the input through the neural network to get its predictions
  optimize.setRate(lR); //sets optimzier leraning rate
  
  float* eValData = evalValue.gpuData(); //easy pointesr for all the gpu data
  float* ePolData = evalPolicy.gpuData();
  float* cValData = correctValue.gpuData();
  float* cPolData = correctPolicy.gpuData();
  const int sV = evalValue.size, sP = evalPolicy.size, thrdCnt = 256; //constant variables for defining the gpu kernel calls
  dim3 gridDimV((sV + thrdCnt - 1) / thrdCnt), gridDimP((sP + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt); //same
  Tensor vGrad(evalValue.dimensions, TensorLocation::GPU, evalValue.n); //value head gradient tensor
  Tensor pGrad(evalPolicy.dimensions, TensorLocation::GPU, evalPolicy.n); //policy head gradient tensor
  ValueLossKernel<<<gridDimV, blockDim>>>(eValData, cValData, vGrad.gpuData(), vGrad.size); //kernel launches
  PolicyLossKernel<<<gridDimP, blockDim>>>(ePolData, cPolData, pGrad.gpuData(), pGrad.size);
  this->backPropagate(vGrad, pGrad);
  std::pair<std::vector<Tensor*>, std::vector<Tensor*>> trainTensors; //variable holding trainable tensors and their batched gradients to perform optimization
  for(auto& ptr : body){ //initial body section loop
    auto* layer = ptr.get();
    auto [Main, Grad] = layer->getLearningData();
    trainTensors.first.insert(trainTensors.first.end(), Main.begin(), Main.end());
    trainTensors.second.insert(trainTensors.second.end(), Grad.begin(), Grad.end());
  }
  for(auto& ptr : policy){ //policy head loop
    auto* layer = ptr.get();
    auto [Main, Grad] = layer->getLearningData();
    trainTensors.first.insert(trainTensors.first.end(), Main.begin(), Main.end());
    trainTensors.second.insert(trainTensors.second.end(), Grad.begin(), Grad.end());
  }
  for(auto& ptr : value){ //value head loop
    auto* layer = ptr.get();
    auto [Main, Grad] = layer->getLearningData();
    trainTensors.first.insert(trainTensors.first.end(), Main.begin(), Main.end());
    trainTensors.second.insert(trainTensors.second.end(), Grad.begin(), Grad.end());
  }
  optimize.batchOptimize(trainTensors);
  ThreadControl::cout(std::string("Optimized"));
  auto elapsed = std::chrono::steady_clock::now() - start;
  std::cout<<std::string("Time in back flow: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  


}
//this evaulate is currently designed so that its only used by the class so inp can be moved and destroyed
//also assumes each batched tensor was its own batch of 1 so there are as many promises as batch dimensions
void NeuralNetwork::batchEvaluate(std::vector<std::unique_ptr<Request>>& r, Tensor& inp, bool train){
  auto start = std::chrono::steady_clock::now();
  
  int i = 0;
  for(auto& layer : this->body){
    auto start = std::chrono::steady_clock::now();
    inp = layer->forward(inp, train); //passes tensor through main body layers
    auto elapsed = std::chrono::steady_clock::now() - start;
    //std::cout<<std::string("Time in layer ") + std::to_string(i) + std::string(": ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
    i++;
  }

  auto elapsed = std::chrono::steady_clock::now() - start;
  ////std::cout<<std::string("Time in body: ") + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

  Tensor P(std::move(inp)); //fast move since inp is assumed to be no longer used
  Tensor V(P); //deep copies for split head
  i = 0;
  for(auto& layer : this->policy){ //iterates through the policy head layers
    auto start = std::chrono::steady_clock::now();
    P = layer->forward(P, train);
    auto elapsed = std::chrono::steady_clock::now() - start;
    //std::cout<<std::string("Time in policy layer ") + std::to_string(i) + std::string(": ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl; 
    i++;
  }
  i=0;
  elapsed = std::chrono::steady_clock::now() - start;
  //ThreadControl::cout(std::string("To policy: ") + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));

  for(auto& layer : value){ //iterates through the value head layers
    auto start = std::chrono::steady_clock::now();
    V = layer->forward(V, train);
    auto elapsed = std::chrono::steady_clock::now() - start;
    //std::cout<<std::string("Time in value layer ") + std::to_string(i) + std::string(": ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl; 
    i++;
  }
  elapsed = std::chrono::steady_clock::now() - start;
  //ThreadControl::cout(std::string("To value: ") + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));
  start = std::chrono::steady_clock::now();
  i = 0;
  for(auto& requestPtr : r){
    auto* request = requestPtr.get();
    P.cpuSend();
    request->pPromise.set_value(P.segment(i)); //fulfills each individual tensors promise from the batch
    request->vPromise.set_value(V.segment(i));
    i++;
  }
  elapsed = std::chrono::steady_clock::now() - start;
  return;
}

void NeuralNetwork::evaluationLoop(){
  auto start = std::chrono::steady_clock::now();
  auto startI = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::steady_clock::now() - start;
  this->run = true;
  const auto batchTime = std::chrono::milliseconds(10);
  
  //std::cout<<"In"<<std::endl;

  while(this->run){
    std::vector<std::unique_ptr<Request>> requestBatch;
    std::vector<Tensor*> tensorBatch;
    start = std::chrono::steady_clock::now();
    while(elapsed < batchTime && requestBatch.size() < this->batchSize){ //waits for a certain amount of inputs or a max time before batching them for evaluation
      std::lock_guard<std::mutex> lock(this->gQueueMutex); //blocks until it can lock the queue, adds the first entry, then unlocks it
      if(!this->queue.empty()){
        requestBatch.push_back(nullptr);
        requestBatch[requestBatch.size()-1].swap(this->queue.front());
        tensorBatch.push_back(&requestBatch[requestBatch.size()-1]->inpState);
        this->queue.pop();
      }
      else{
        std::this_thread::sleep_for(std::chrono::microseconds(100)); //blocks for a little if queue is empty
      }
      elapsed = std::chrono::steady_clock::now() - start;
    }
    if(requestBatch.size() != 0){
      Tensor inputBatch;
      start = std::chrono::steady_clock::now();
      inputBatch.batchBuild(tensorBatch, TensorLocation::GPU);
      
      cudaDeviceSynchronize;
      elapsed = std::chrono::steady_clock::now() - start;
      //ThreadControl::cout(std::string("Time to batch: ") + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));

      this->batchEvaluate(requestBatch, inputBatch); //sends the batch to be evaluated, fulfills promises
      
      //ThreadControl::cout(std::to_string(tensorBatch.size()));
      run = false;
    }
    else{
    }
  }
}

void NeuralNetwork::evaluationRequest(std::unique_ptr<Request>& req){
  Request* r = req.get();
  std::lock_guard<std::mutex> lock(this->gQueueMutex);
  this->queue.push(nullptr);
  this->queue.back().swap(req);
}

void NeuralNetwork::shutDown(){
  this->run = false;
}