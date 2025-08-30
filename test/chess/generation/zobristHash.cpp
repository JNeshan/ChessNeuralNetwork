#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <vector>
#include <random>
#include <fstream>
#include <set>

//file is used to generate zobrist hash numbers like magic numbers
//does not need to be ran again

enum PieceType {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE};
enum Color {WHITE, BLACK, NO_COLOR};

const int numPieceTypes = 6; 
const int numColors = 2;    
const int numSquares = 64;
const int numCastles = 4;
const int numFiles = 8;

inline int get_zobrist_piece_index(PieceType type, Color color) {
    if (type == NO_PIECE_TYPE || color == NO_COLOR) return -1; 
    return static_cast<int>(type) + (static_cast<int>(color) * numPieceTypes);
}

struct zobristKeys {
    std::array<std::array<uint64_t, numSquares>, numPieceTypes * numColors> pieceSquare;
    uint64_t blackToMove;
    std::array<uint64_t, numCastles> castlingRights;
    std::array<uint64_t, numFiles> enpassantFile;

    zobristKeys() : blackToMove(0) {
      for (auto& arr_sq : pieceSquare) arr_sq.fill(0);
      castlingRights.fill(0);
      enpassantFile.fill(0);
    }
};

std::vector<uint64_t> serializeKeys(const zobristKeys& keys) {
  std::vector<uint64_t> serialized;
  serialized.reserve( (numPieceTypes * numColors * numSquares) + 1 + numCastles + numFiles);

  for (const auto& piece_array : keys.pieceSquare) {
    for (uint64_t key : piece_array) {
      serialized.push_back(key);
    }
  }
  serialized.push_back(keys.blackToMove);
  for (uint64_t key : keys.castlingRights) {
    serialized.push_back(key);
  }
  for (uint64_t key : keys.enpassantFile) {
    serialized.push_back(key);
  }
  return serialized;
}

using namespace std;
int main(){
  std::mt19937_64 generate(1234567890ULL);

  array<array<uint64_t, 64>, 12> pieceSquares;
  uint64_t blackToMove;
  array<uint64_t, 4> castlingRights;
  array<uint64_t, 8> enpassantFile;

  for(int p = 0; p < 6 * 2;  p++){
    for(int pos = 0; pos < 64; pos++){
      pieceSquares[p][pos] = generate();
    }
  }
  blackToMove = generate();
  for(int i = 0; i < 4; i++){
    castlingRights[i] = generate();
  }
  for(int i = 0; i < 8; i++){
    enpassantFile[i] = generate();
  }
  zobristKeys keys;
  keys.blackToMove = blackToMove;
  keys.castlingRights = castlingRights;
  keys.enpassantFile = enpassantFile;
  keys.pieceSquare = pieceSquares;
  vector<uint64_t> storedKeys = serializeKeys(keys);
  
  std::ofstream output("zobristTable.hpp");
  output << "#include<vector>\n#include<cstdint>\nstatic const std::vector<uint64_t> zobristTable = {\n";
  for(int i = 0; i < storedKeys.size(); i++){
    output << storedKeys[i];
    if(storedKeys.size()-i != 1)
       output << "ULL, ";
  }
  output <<"ULL\n};";
  set<uint64_t> unique(storedKeys.begin(), storedKeys.end());
  if(unique.size() != storedKeys.size()){ 
    cout<<"bad seed"<<endl;
  }
  else{
    cout<<"good"<<endl;
  }

}