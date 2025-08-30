#include <iostream>
#include <fstream>
#include "header/tensor.h"
#include "header/matriceMath.h"
#include "header/generator.h"
#include "header/optimizer.h"
#include "header/layer.h"
#include "header/denseLayer.h"
#include "header/reluLayer.h"
#include "header/convolutionLayer.h"
#include "header/tanhLayer.h"
#include "header/softmaxLayer.h"
#include "header/normalizationLayer.h"
#include "header/mcts.h"


using namespace std;
using TL = TensorLocation;

int main(){
  bool generate = true;
  std::vector<Layer*> body, policyHead, valueHead;
  Layer* l = nullptr;
  l = new ConvolutionLayer(256, 17, 3, 3);
  body.push_back(l);
  l = new NormalizationLayer(true, 256);
  body.push_back(l);
  l = new ReLULayer;
  body.push_back(l);

  for(int i = 0; i < 5; i++){
    l = new ConvolutionLayer(256, 256, 3, 3);
    body.push_back(l);
    l = new NormalizationLayer(true, 256);
    body.push_back(l);
    l = new ConvolutionLayer(256, 256, 3, 3);
    body.push_back(l);
    l = new NormalizationLayer(true, 256);
    body.push_back(l);
    l = new ReLULayer;
    body.push_back(l);
  }


  std::ofstream outF("Output.txt");
  if(!outF.is_open()){
    cout<<"failed to open"<<endl;
    return -1;
  }
  std::string binF = "tensors.bin";
  std::ofstream outBin;
  std::ifstream inBin;
  Tensor T({2, 2}, TensorLocation::GPU, 2);
  Generator::tGen(T);
  Tensor O({2, 2});
  T.batch(O);
  T.cpuSend();
  T.batch(O);
  T.writeTensor(outF);
  
  outF.close();
  outBin.close();
  cout<<"ran"<<endl;
  return 0;
}