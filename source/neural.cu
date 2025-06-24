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
  Tensor T(input);
  for(int i = 0; i < layers.size(); i++){
    if(layTypes[i] == LayerType::CON){
      int n = T.dimensions[0]; int c = T.size / (n * 64);
      T.reshape({n, c, 8, 8});
    }
    else if(layTypes[i] == LayerType::DENSE){
      int n = T.dimensions[0]; int f = T.size / n;
      T.reshape({n, f});
    }
    else if(layTypes[i] == LayerType::RELU){
      int n = T.dimensions[0];
      std::vector<int> dim(4 - T.dimensions.size(), 1);
      for(auto iT : T.dimensions){
        dim.push_back(iT);
      }
      T.reshape(dim);
    }
    auto *li = layers[i].get();
    Tensor result(li->forward(T));
  }
}

