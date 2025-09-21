#include "../header/tensorization.h"
#include "../header/generator.h"
/*
Each of blacks piece types 6
Each of whites piece types 12
Player to move
Castling Rights
En Passant
*/
Tensorization::Tensorization(){}

Tensor Tensorization::tensorize(const chessState& state){
  Tensor T({1, 17, 8, 8}, TensorLocation::CPU);
  float* data = T.cpuData();
  int p = 0;
  std::array<uint64_t, 6> whiteBitboard = state.bitboards[0], blackBitboard = state.bitboards[1];
  while(p < 6){
    for(int i = 0; i < 64; i++){
      if(whiteBitboard[p] & 1ULL << i){
        data[p * 64 + i] = 1.0f;
      }
    }
    p++;
  }
  while(p < 12){
    for(int i = 0; i < 64; i++){
      if(blackBitboard[p-6] & 1ULL << i){
        data[p * 64 + i] = 1.0f;
      }
    }
    p++;
  }
  while(p < 16){
    if(state.castleState & 1ULL << (p-12)){
      for(int i = 0; i < 64; i++){
        data[p * 64 + i] = 1.0f;
      }
    }
    p++;
  }
  if(state.enpassant != 0ULL){
    int pos = __builtin_ctzll(state.enpassant);
    data[p * 64 + pos] = 1.0f;
  }
  return T;
}



