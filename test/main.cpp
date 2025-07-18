#include <iostream>
#include <fstream>
#include "header/tensor.h"
#include "header/matriceMath.h"
#include "header/generator.h"
#include "header/optimizer.h"
#include "header/layer.h"
#include "header/denseLayer.h"



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
  dense.genTensorData();
  inBin = std::ifstream(binF);
  //dense.loadTensor(inBin);
  Tensor T({2, 2}, TL::GPU);
  Generator::tGen(T);
  Tensor O = dense.forward(T);
  inBin.close();
  T.writeTensor(outF);
  dense.cleanSave(outF);
  O.writeTensor(outF);
  outF.close();
  outBin.close();
  cout<<"ran"<<endl;
  return 0;
}