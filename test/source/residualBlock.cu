#include "../header/residualBlock.h"


ResidualBlock::ResidualBlock(){
  Layer* l = layers[0].get();
  this->layers.push_back(std::make_unique<ConvolutionLayer>(256, 256, 3, 3));
  this->layers.push_back(std::make_unique<NormalizationLayer>(true, 256));
  this->layers.push_back(std::make_unique<ReLULayer>());
  this->layers.push_back(std::make_unique<ConvolutionLayer>(256, 256, 3, 3));
  this->layers.push_back(std::make_unique<NormalizationLayer>(true, 256));
  this->layers.push_back(std::make_unique<ConvolutionLayer>(256, 256, 3, 3));
  this->layers.push_back(std::make_unique<ReLULayer>());
}

Tensor ResidualBlock::forward(const Tensor& T, bool train){
  this->inp = Tensor(T);
  Tensor iT = Tensor(T);
  for(int i = 0; i < this->layers.size(); i++){
    Layer* l = layers[i].get();
    iT = l->forward(iT, true);
  }
  iT.gpuAdd(this->inp);
  return iT;
}

Tensor ResidualBlock::backward(const Tensor& gradient){
  this->inpGrad = Tensor(gradient);
  for(int i = layers.size() - 1; i >= 0; i--){
    Layer* l = layers[i].get();
    this->inpGrad = l->backward(this->inpGrad);
  }
  return this->inpGrad;
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

