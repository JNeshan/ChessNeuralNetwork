#include "../header/threadControl.h"

std::mutex ThreadControl::gs_ConsoleMutex; 

std::string ThreadControl::cin(){
  //std::unique_lock<std::mutex> lock(ThreadControl::gs_ConsoleMutex);
  std::string out;
  std::cin >> out;
  return out;
}

void ThreadControl::cout(std::string out){
  std::unique_lock<std::mutex> lock(ThreadControl::gs_ConsoleMutex);
  std::cout<<out<<std::endl;
}