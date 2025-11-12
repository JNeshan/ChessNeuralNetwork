#include "header/tensor.cuh"
#include "header/generator.h"
#include <iostream>

using namespace std;

int main(){

  Tensor<float> x;
  x = Tensor<float>({2, 2, 2, 2}, TensorLocation::CPU, 4);
  cout<<x.at(2)<<endl;
  cout<<x.at(1, 1, 1, 1)<<endl;
  cout<<x.cpuData()<<endl;
  x.gpuSend();
  Generator gen;
  gen.vGen(x.size, 1, x.gpuData());
  Tensor<__half> y;
  try
  {
    y = x;
    cout<<__half2float(y.at(2))<<endl;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  
  try
  {
    x = y;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  try
  {
    x += y;
    cout<<__half2float(x.at(2))<<endl;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  
  try
  {
    Tensor<__half> z = x + y;
    cout<<__half2float(z.at(2))<<endl;

  }
  catch(const runtime_error& e)
  {
    cerr << e.what() << '\n';
  }
  

  return 0;
}