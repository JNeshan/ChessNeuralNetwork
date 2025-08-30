//zobristKeys.h

#ifndef ZOBRISTKEYS_H
#define ZOBRISTKEYS_H
#include "zobristTable.hpp"
#include <vector>
#include <array>
#include <cstdint>

enum PieceType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NOPIECE };
enum Color { WHITE, BLACK, NOCOLOR };

const int numPieceTypes = 6, numColors = 2, numSquares = 64, numCastles = 4, numFiles = 8;

struct zobristKeys {
  std::array<std::array<uint64_t, numSquares>, numPieceTypes * numColors> pieceSquare;
  uint64_t blackToMove;
  std::array<uint64_t, numCastles> castlingRights;
  std::array<uint64_t, numFiles> enpassantFile;

  zobristKeys(){
    deserialize();
  }

  void deserialize(){
    int index = 0/*, size = (numPieceTypes * numColors * numSquares) + 1 + numCastles + numFiles*/;
    for(int i = 0; i < numPieceTypes * numColors; i++){
      for(int pos = 0; pos < numSquares; pos++){
        pieceSquare[i][pos] = zobristTable[index];
        index++;
      }
    }
    blackToMove = zobristTable[index];
    index++;
    for(int i = 0; i < numCastles; i++){
      castlingRights[i] = zobristTable[index];
      index++;
    }
    for(int i = 0; i < numFiles; i++){
      enpassantFile[i] = zobristTable[index];
    }
  }
};


#endif