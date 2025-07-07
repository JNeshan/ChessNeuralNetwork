#include "header/neural.h"
#include "cuda_runtime.h"


struct layerC{
  std::vector<std::unique_ptr<Layer>> layers;


};

NeuralNetwork::NeuralNetwork(std::string fName){
  ready = true;
  file = fName;
}

void NeuralNetwork::SetLayers(const std::vector<std::unique_ptr<Layer>>& l, const std::vector<LayerType>& lT){
  layers = l;
  layTypes = lT;
  if(!rF.good()){
    for(int i = 0; i < l.size(); i++){
      auto li = l[i].get();
      li->genTensorData();
    }
  }
  else{
    if(!ready){
      throw("File left open");
    }
    rF.open(file);
    bool ready = false;
    for(int i = 0; i < l.size(); i++){
      auto li = l[i].get();
      li->loadTensor(rF);
    }
    rF.close();
    ready = true;
  }
}

void NeuralNetwork::RunNetwork(Tensor& input){
  
}

