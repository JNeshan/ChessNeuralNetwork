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

NeuralNetwork::NeuralNetwork(std::vector<Layer*>& b, std::vector<Layer*>& pH, std::vector<Layer*>& vH) : body(b), policy(pH), value(vH), optimize(1){}

std::pair<Tensor, Tensor> NeuralNetwork::evaluate(const Tensor& inp, bool train){
  Tensor T = inp;
  T.gpuSend();
  for(auto& layer : body){
    T = layer->forward(T, train);
  }
  Tensor P = policy[0]->forward(T, train);
  Tensor V = value[0]->forward(T, train);
  for(int i = 1; i < policy.size(); i++){
    P = policy[i]->forward(P, train);
  }
  for(int i = 1; i < value.size(); i++){
    V = value[i]->forward(V, train);
  }
  return std::make_pair(P, V);
}

void NeuralNetwork::backPropagate(const Tensor& v, const Tensor& p){
  Tensor V = value[value.size()-1]->backward(v);
  Tensor P = policy[policy.size()-1]->backward(p);
  for(int i = value.size()-2; i >= 0; i--){
    V = value[i]->backward(V);
  }
  for(int i = policy.size()-2; i >= 0; i--){
    P = policy[i]->backward(P);
  }
  V.gpuAdd(P);
  for(int i = body.size()-1; i >= 0; i--){
    V = body[i]->backward(V);
  }
}

void NeuralNetwork::train(const Tensor& inp, const Tensor& correctValue, const Tensor& correctPolicy, const int lR){
  auto [evalValue, evalPolicy] = this->evaluate(inp, true); //runs the input through the neural network to get its predictions
  optimize.setRate(lR); //sets optimzier leraning rate
  float* eValData = evalValue.gpuData(); //easy pointesr for all the gpu data
  float* ePolData = evalPolicy.gpuData();
  float* cValData = correctValue.gpuData();
  float* cPolData = correctPolicy.gpuData();
  const int sV = evalValue.size, sP = evalPolicy.size, thrdCnt = 256; //constant variables for defining the gpu kernel calls
  dim3 gridDimV((sV + thrdCnt - 1) / thrdCnt), gridDimP((sP + thrdCnt - 1) / thrdCnt), blockDim(thrdCnt);
  Tensor vGrad(evalValue.dimensions, TensorLocation::GPU, evalValue.n);
  Tensor pGrad(evalPolicy.dimensions, TensorLocation::GPU, evalPolicy.n);
  ValueLossKernel<<<gridDimV, blockDim>>>(eValData, cValData, vGrad.gpuData(), vGrad.size);
  PolicyLossKernel<<<gridDimP, blockDim>>>(ePolData, cPolData, pGrad.gpuData(), pGrad.size);
  this->backPropagate(vGrad, pGrad);
  std::pair<std::vector<Tensor*>, std::vector<Tensor*>> trainTensors;
  
  for(auto layer : body){
    auto [Main, Grad] = layer->getLearningData();
    trainTensors.first.insert(trainTensors.first.end(), Main.begin(), Main.end());
    trainTensors.second.insert(trainTensors.second.end(), Grad.begin(), Grad.end());
  }
  for(auto layer : policy){
    auto [Main, Grad] = layer->getLearningData();
    trainTensors.first.insert(trainTensors.first.end(), Main.begin(), Main.end());
    trainTensors.second.insert(trainTensors.second.end(), Grad.begin(), Grad.end());
  }
  for(auto layer : value){
    auto [Main, Grad] = layer->getLearningData();
    trainTensors.first.insert(trainTensors.first.end(), Main.begin(), Main.end());
    trainTensors.second.insert(trainTensors.second.end(), Grad.begin(), Grad.end());
  }
  optimize.batchOptimize(trainTensors);
}

