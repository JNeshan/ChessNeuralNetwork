#include "../header/residualBlock.h"


ResidualBlock::ResidualBlock(){
  this->layers.push_back(std::make_unique<ConvolutionLayer>(256, 256, 3, 3, 1));
  this->layers.push_back(std::make_unique<NormalizationLayer>(true, 256));
  this->layers.push_back(std::make_unique<ReLULayer>());
  this->layers.push_back(std::make_unique<ConvolutionLayer>(256, 256, 3, 3, 1));
  this->layers.push_back(std::make_unique<NormalizationLayer>(true, 256));
  this->layers.push_back(std::make_unique<ReLULayer>());
}

ResidualBlock::ResidualBlock(const ResidualBlock& lay){
  for(auto& l : lay.layers){
    const auto& lRef = l.get();
    this->layers.push_back(lRef->clone());
  }
}

ResidualBlock::~ResidualBlock(){}

std::unique_ptr<Layer> ResidualBlock::clone(){
  return(std::make_unique<ResidualBlock>(*this));
}

Tensor ResidualBlock::forward(Tensor& T, bool train){
  auto start = std::chrono::steady_clock::now();
  this->inp = Tensor(T);
  for(int i = 0; i < layers.size()-1; i++){
    start = std::chrono::steady_clock::now();
    auto* layer = layers[i].get();
    T = layer->forward(T, train);
    auto elapsed = std::chrono::steady_clock::now() - start;
    //std::cout<<std::string("Time in residual layer ") + std::to_string(i) + std::string(": ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl; 
  }
  T.gpuAdd(this->inp);
  auto* layer = layers[layers.size()-1].get();
  T = layer->forward(T, train);
  auto elapsed = std::chrono::steady_clock::now() - start;
  ////std::cout<<std::string("Time in residual: ") + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())<<std::endl;  
  return std::move(T);
}

Tensor ResidualBlock::backward(Tensor& gradient){
  this->inpGrad = layers[layers.size()-1]->backward(gradient);
  Tensor skipGrad(inpGrad);
  for(int i = layers.size() - 2; i >= 0; i--){
    Layer* l = layers[i].get();
    this->inpGrad = l->backward(this->inpGrad);
  }
  skipGrad.gpuAdd(this->inpGrad);
  return std::move(skipGrad);
}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>> ResidualBlock::getLearningData(){
  std::pair<std::vector<Tensor*>, std::vector<Tensor*>> trainingTensors;
  std::vector<Tensor*>* tensorInp = &trainingTensors.first, *tensorGrad = &trainingTensors.second; //pointers for cleaner operations
  tensorInp->push_back(&this->inp);
  tensorGrad->push_back(&this->inpGrad);
  for(int i = 0; i < this->layers.size(); i++){
    Layer* l = layers[i].get();
    std::pair<std::vector<Tensor*>, std::vector<Tensor*>> tT = l->getLearningData();
    tensorInp->insert(tensorInp->end(), tT.first.begin(), tT.first.end());
    tensorGrad->insert(tensorGrad->end(), tT.second.begin(), tT.second.end());
  }
  return trainingTensors;
}

void ResidualBlock::genTensorData(){
  for(int i = 0; i < this->layers.size(); i++){
    Layer* l = this->layers[i].get();
    l->genTensorData();
  }
}

void ResidualBlock::loadTensor(std::ifstream& iF){
  for(int i = 0; i < this->layers.size(); i++){
    Layer* l = layers[i].get();
    l->loadTensor(iF);
  }
}

void ResidualBlock::saveTensor(std::ofstream& oF){
  for(int i = 0; i < this->layers.size(); i++){
    Layer* l = layers[i].get();
    l->saveTensor(oF);
  }
}

void ResidualBlock::cleanSave(std::ofstream& oF){
  for(int i = 0; i < this->layers.size(); i++){
    Layer* l = layers[i].get();
    l->cleanSave(oF);
  }
}

