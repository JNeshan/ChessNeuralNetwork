#include "../header/flattenLayer.h"

FlattenLayer::FlattenLayer(){}

FlattenLayer::FlattenLayer(const FlattenLayer& lay){}

std::unique_ptr<Layer> FlattenLayer::clone(){
  return(std::make_unique<FlattenLayer>(*this));
}

Tensor FlattenLayer::forward(Tensor& T, bool train){
  this->inpDim = T.dimensions;
  Tensor out(T);
  out.reshape({out.dimensions[0], out.size / out.dimensions[0]}, 2);
  return std::move(out);
}

Tensor FlattenLayer::backward(Tensor& gradient){
  Tensor out(gradient);
  out.reshape(this->inpDim, this->inpDim.size());
  return std::move(out);
}

FlattenLayer::~FlattenLayer(){

}

void FlattenLayer::genTensorData(){

}

void FlattenLayer::loadTensor(std::ifstream& iF){

}

void FlattenLayer::saveTensor(std::ofstream& oF){

}

void FlattenLayer::cleanSave(std::ofstream& oF){

}

std::pair<std::vector<Tensor*>, std::vector<Tensor*>>FlattenLayer::getLearningData(){
  return std::pair<std::vector<Tensor*>, std::vector<Tensor*>>();
}