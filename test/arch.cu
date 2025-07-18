#include <iostream>
#include "cuda_runtime.h"

using namespace std;
int main(){
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    cout<<"fail 1"<<endl;
    return 1;
  }

  if(!count){
    cout<<"No device"<<endl;
    return 1;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties_v2(&deviceProp, 0);
  cout<<deviceProp.major<<"."<<deviceProp.minor<<endl;
  return 0;
}