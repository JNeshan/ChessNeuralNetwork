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
  Tensor T({1, 8, 8, 17}, TensorLocation::CPU); //NHWC format
  float* data = T.cpuData();
  std::array<uint64_t, 6> whiteBitboard = state.bitboards[0], blackBitboard = state.bitboards[1];

  for(int pos = 0; pos < 64; pos++){
    int p = 0;
    while(p < 6){
      if(whiteBitboard[p] & 1ULL << pos){
        data[pos * 17 + p] = 1.0f;
      }
      if(blackBitboard[p] & 1ULL << pos){
        data[pos * 17 + p + 6] = 1.0f;
      }
      p++;
    }
    p = 12;
    while(p < 16){
      if(state.castleState & 1ULL << (p - 12)){
        data[pos * 17 + p] = 1.0f;
      }
      p++;
    }
}
  if(state.enpassant != 0ULL){
    int pos = __builtin_ctzll(state.enpassant);
    data[pos * 17 + 16] = 1.0f;
  }
  return T;
}



