#ifndef THREADCONTROL_H
#define THREADCONTROL_H

#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>

class ThreadControl{

public:
  ThreadControl();
  static std::string cin();
  static void cout(std::string out);
  static std::mutex gs_ConsoleMutex;

private:

};

#endif