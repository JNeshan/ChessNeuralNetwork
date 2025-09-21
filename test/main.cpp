#include <iostream>
#include <fstream>
#include "header/chessAi.h"
#include <thread>
#include <future>



using namespace std;
using TL = TensorLocation;

int main(){

  ChessAI AI;
  cout<<"Main run"<<endl;
  //std::ofstream oF("output.txt");
  AI.generateValues();
  //AI.saveLayers(oF);
  //oF.close();
  std::ifstream iF("output.txt");
  AI.loadLayers(iF);
  
  AI.train();
  std::cout<<"Training done"<<std::endl;
  iF.close();
  return 0;
}