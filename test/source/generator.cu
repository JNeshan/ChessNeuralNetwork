#include "../header/generator.h"

__inline__ void TryCuda(curandStatus_t err){
  if(err != CURAND_STATUS_SUCCESS){
    fprintf(stderr, "curand Error in %s at line %d: %s\n", __FILE__, __LINE__, (char)('0' + err));
      exit(EXIT_FAILURE);
  }
}

__inline__ curandGenerator_t createCurand(){
  curandGenerator_t gen;
  TryCuda(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  TryCuda(curandSetPseudoRandomGeneratorSeed(gen, 0));
  TryCuda(curandSetGeneratorOffset(gen, 1));
  TryCuda(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT));
  return gen;
}



curandGenerator_t Generator::cGen = createCurand();

Generator::Generator(){
  int a = 0;
}

void Generator::tGen(const Tensor& T){
  float* output = T.gpuData();
  TryCuda(curandGenerateUniform(cGen, output, T.size));
  return;
}

void Generator::dGen(const int s, float* data){
  TryCuda(curandGenerateUniform(cGen, data, s));
}