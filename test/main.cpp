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



using namespace std;
using TL = TensorLocation;

int main(){

  std::ofstream outF("Output.txt");
  if(!outF.is_open()){
    cout<<"failed to open"<<endl;
    return -1;
  }
  std::string binF = "tensors.bin";
  std::ofstream outBin;
  std::ifstream inBin;

  //MatriceMath cl;
  DenseLayer dense(2, 2);
  ReLULayer relu;
  ConvolutionLayer convolution(8, 2, 1, 1);
  tanhLayer tanh;
  SoftmaxLayer softmax;
  //Tensor T({4, 4}, TensorLocation::GPU);
  //Generator::tGen(T);
  //Tensor O = tanh.forward(T);
  //T.writeTensor(outF);
  //O.writeTensor(outF);
  Tensor T({1, 4672}, TensorLocation::GPU);
  Generator::tGen(T);
  Tensor O = softmax.forward(T);
  T.writeTensor(outF);
  O.writeTensor(outF);
  outF.close();
  outBin.close();
  cout<<"ran"<<endl;
  return 0;
}