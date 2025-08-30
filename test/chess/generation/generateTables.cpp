 /*
This file does not need to be executed, including it in additional files causes conflict
This file is included to demonstrate my process to justify the use of a precomputed array
since everything remains consistent, it is only necessary for this to ever be run once so it is not used with the program
It can be rerun freely to check its validity but the generation might take a minute
*/

#include <array>
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <fstream>
#include <random>

inline int position(int rank, int file){ //converts rank and file to single int position
  return rank * 8 + file;
}
//random number generation for perfect hash
static std::mt19937_64 rng64{ std::random_device{}() }; 
static inline uint64_t random_u64() {
    return rng64();
}
uint64_t random_sparse() {
  return random_u64() & random_u64() & random_u64();
}
//movement vectors to generate each pieces masks
const std::vector<std::pair<int,int>> knightDir = {
  {2, 1}, {1, 2},
  {-1, 2}, {-2, 1},
  {-2, -1}, {-1, -2},
  {1, -2}, {2, -1}
};
const std::vector<std::pair<int,int>> bishopDir = {
  {1, 1}, {1, -1},
  {-1, 1}, {-1, -1}
};
const std::vector<std::pair<int,int>> queenDir = {
  {1,  0}, {-1,  0}, {0, 1}, {0, -1},
  {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
};
const std::vector<std::pair<int,int>> rookDir = {
  {1,  0}, {-1,  0},
  {0, 1}, {0, -1}
};
const std::vector<std::pair<int,int>> kingDir = {
  {1,  0}, {-1,  0}, {0, 1}, {0, -1},
  {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
};
//generates each unique permutation of a blocker mask to represent every relevant board state for the piece
//allied and enemy pieces are not discriminated
std::vector<uint64_t> generateBlockerBoards(int square, uint64_t mask) {
  std::vector<uint64_t> blockers;
  int bits = __builtin_popcountll(mask);
  int variations = 1 << bits; //binary permutations naturally means 2^bits
  for(int i = 0; i < variations; i++){ //loops for every possible variations, max of 2^12 for rooks and 2^9 for bishops
    uint64_t blocker = 0ULL; //initializes empty board state representation to add
    int bitIndex = 0; 
    for(int j = 0; j < 64; j++){ 
      if(mask & (1ULL << j)){ 
        if(i & (1 << bitIndex))
          blocker |= (1ULL << j); 
        bitIndex++;
      }
    }
    blockers.push_back(blocker);
  }
  return blockers;
}
//generates the base blocker masks for rooks, end of board is ignored since its uneccessary
uint64_t rookMask(int pos){
  uint64_t mask = 0ULL; 
  int r = pos / 8;
  int f = pos % 8;

  for(auto [dR, dF] : rookDir){
    if(dR != 0){
      int newR = r + dR;
      while(newR > 0 && newR < 7){
        mask |= 1ULL << position(newR, f);
        newR += dR;
      }
    }
    else{
      int newF = f + dF;
      while(newF > 0 && newF < 7){
        mask |= 1ULL << position(r, newF);
        newF += dF;
      }
    }
  }
  return mask;
}
//generates base blocker masks for bishops, same logic different movement vector
uint64_t bishopMask(int pos){ 
  uint64_t mask = 0ULL;
  int r = pos / 8;
  int f = pos % 8;

  for(auto [dR, dF] : bishopDir){
    int newR = r + dR, newF = f + dF;
    while(newR > 0 && newR< 7 && newF > 0 && newF < 7){
      mask |= 1ULL << position(newR, newF);
      newR += dR; newF += dF;
    }
  }

  return mask;
}
//generates every possible attack board based on every blocker board for a mask, sets every space up to and including the first blocker for a direction to 1
std::vector<uint64_t> rookAttacksFind(int pos, std::vector<uint64_t> blockerList){
  std::vector<uint64_t> attackList;
  int rank = pos / 8;
  int file = pos % 8;
  for(int i = 0; i < blockerList.size(); i++){
    uint64_t attacks = 0ULL;
    auto blockers = blockerList[i];
    for(auto [dR, dF] : rookDir){
      int newR = rank, newF = file;
      newR += dR; newF += dF;
      while(newR >= 0 && newR <= 7 && newF >= 0 && newF <= 7){
        int newPos = position(newR, newF);
        attacks |= 1ULL << newPos;
        if(blockers & (1ULL << newPos)){
          break;
        }
        newR += dR; newF += dF;
      }
    }
    attackList.push_back(attacks);
  }
  return attackList;
}
//same logic is rook
std::vector<uint64_t> bishopAttacksFind(int pos, std::vector<uint64_t> blockerList){
  std::vector<uint64_t> attackList;
  int rank = pos / 8;
  int file = pos % 8;
  for(int i = 0; i < blockerList.size(); i++){
    uint64_t attacks = 0ULL;
    auto blockers = blockerList[i];
    for(auto [dR, dF] : bishopDir){
      int newR = rank, newF = file;
      newR += dR; newF += dF;
      while(newR >= 0 && newR <= 7 && newF >= 0 && newF <= 7){
        
        int newPos = position(newR, newF);
        attacks |= 1ULL << newPos;
        if(blockers & (1ULL << newPos)){
          break;
        }
        newR += dR; newF += dF;
      }
    }
    attackList.push_back(attacks);
  }
  return attackList;
}
//retrieves the number of 1 bits in the 64 bit attack mask, needed for indexing
int countRelevantBits(uint64_t attackMask) {
  return __builtin_popcountll(attackMask);  // GCC/Clang
}
//checks ifthe magic number is a perfect hash
bool isMagicValid(uint64_t magic, std::vector<uint64_t>& blockers, std::vector<uint64_t>& attacks, uint64_t mask, int relevantBits){
  std::unordered_map<int, uint64_t> used; //creates a map to check if a hash creates a nonunique index
  for(size_t i = 0; i < blockers.size(); i++){
    uint64_t index = ((blockers[i] & mask) * magic) >> (64 - relevantBits); //applys magic index formula to test the random number
    if(used.count(index)){ //magic number does not work for the perfect hash if it creates duplicate indexes for different masks
      if(used[index] != attacks[i])
      return false;
    } 
    else{
      used[index] = attacks[i];
    }
  }
  return true;
}
//attack masks for knight, king, and pawn are just relative to position, with pawn also being relative to piece type
std::array<uint64_t, 64> knightMask(){  
  std::array<uint64_t, 64> knightAttacks;
  knightAttacks.fill(0ULL);
  for(int pos = 0; pos < 64; pos++){
    int r = pos / 8, f = pos % 8;
    for(auto [dR, dF] : knightDir){
      int newR = r + dR, newF = f + dF;
      if(newR >= 0 && newR < 8 && newF >= 0 && newF < 8){
        knightAttacks[pos] |= 1ULL << position(newR, newF);
      }
    }
  }
  return knightAttacks;
}

std::array<uint64_t, 128> pawnMask(){
  std::array<uint64_t, 128> pawnAttacks;
  pawnAttacks.fill(0ULL);
  for(int pos = 0; pos < 64; pos++){
    int file = pos % 8;
    if(pos < 56) {
      if(file < 7)
        pawnAttacks[pos] |= 1ULL << (pos + 9);
      if(file > 0)
        pawnAttacks[pos] |= 1ULL << (pos + 7);
    }
    if(pos >= 8){
      if(file < 7)
        pawnAttacks[pos + 64] |= 1ULL << (pos - 7);
      if(file > 0)
        pawnAttacks[pos + 64] |= 1ULL << (pos - 9);
    }
  }
  
  return pawnAttacks;
}

std::array<uint64_t, 64> kingMask(){
  std::array<uint64_t, 64> kingAttacks;
  kingAttacks.fill(0ULL);
  for(int pos = 0; pos < 64; pos++){
    int r = pos / 8, f = pos % 8;
    for(auto [dR, dF] : kingDir){
      int newR = r + dR, newF = f + dF;
      if(newR >= 0 && newR < 8 && newF >= 0 && newF < 8){
        kingAttacks[pos] |= 1ULL << position(newR, newF);
      }
    }
  }
  return kingAttacks;
}

int main(){
  std::ofstream file("magicNumbers.hpp");
  
  std::vector<std::vector<uint64_t>> rookBlockers; //stores each blocker permutation
  std::vector<std::vector<uint64_t>> bishopBlockers;
  std::vector<std::vector<uint64_t>> rookAttackArray; //stores the final attack mask
  std::vector<std::vector<uint64_t>> bishopAttackArray;

  std::vector<int> rookBits; //stores bits used for magic hash
  std::vector<int> bishopBits;
  std::vector<uint64_t> rookMagic; //stores magic numbers for perfect hash
  std::vector<uint64_t> bishopMagic;
  std::vector<uint64_t> rookMaskArray; //stores position masks
  std::vector<uint64_t> bishopMaskArray; 

  std::unordered_map<uint64_t, uint64_t> blockerToAttack;

  for(int pos = 0; pos < 64; pos++){ //iterates through and generates all masks for all positions, stores values needed for indexing in vectors to be outputted    
    uint64_t mask = rookMask(pos);  
    rookMaskArray.push_back(mask);
    std::vector<uint64_t> attackMasks;

    rookBlockers.push_back(generateBlockerBoards(pos, mask));

    int bits = countRelevantBits(mask);

    rookBits.push_back(bits);

    attackMasks = rookAttacksFind(pos, rookBlockers[pos]);

    uint64_t magic = 0ULL; bool found = false;
    while(!found){ //generates random numbers until a magic number that can represent unique positions for each element is found, perfect hash
      magic = random_sparse(); //generates random uint using inlined function
      found = isMagicValid(magic, rookBlockers[pos], attackMasks, mask, bits); //checks for perfect hash
    }
    rookAttackArray.push_back(std::vector<uint64_t>(1 << bits)); //initializes new element with necessary size since indexing is random
    std::cout<<magic<<std::endl; 
    for(int i = 0; i < attackMasks.size(); i++){
      uint32_t magicInd = ((mask & rookBlockers[pos][i]) * magic) >> (64 - bits);
      rookAttackArray[pos][magicInd] = attackMasks[i]; //assigns elements to the output array based on hash
    }

    rookMagic.push_back(magic); //pushes perfect hash magic number to vector
  }
  file << "#include <cstdint>\n\nstatic const uint64_t rookMagicNums[64] = {\n  ";
  for(int i = 0; i < 64; i++){
    file << rookMagic[i];
    if(i + 1 < 64) file  << "ULL, ";
  }
  file << "ULL};\n";

  for(int pos = 0; pos < 64; pos++){ //same loop for bishop masks    
    uint64_t mask = bishopMask(pos);
    bishopMaskArray.push_back(mask);
    std::vector<uint64_t> attackMasks;

    bishopBlockers.push_back(generateBlockerBoards(pos, mask));

    int bits = countRelevantBits(mask);

    bishopBits.push_back(bits);

    attackMasks = bishopAttacksFind(pos, bishopBlockers[pos]);

    uint64_t magic = 0ULL; bool found = false;
    while(!found){ //generates random numbers until a magic number that can represent unique positions for each element is found, perfect hash
      magic = random_sparse(); //generates random uint using inlined function
      found = isMagicValid(magic, bishopBlockers[pos], attackMasks, mask, bits); //checks for perfect hash
    }
    std::cout<<magic<<std::endl;
    bishopAttackArray.push_back(std::vector<uint64_t>(1 << bits)); //initializes new element with necessary size since indexing is random
    for(int i = 0; i < attackMasks.size(); i++){
      uint32_t magicInd = ((mask & bishopBlockers[pos][i]) * magic) >> (64 - bits);
      bishopAttackArray[pos][magicInd] = attackMasks[i]; //assigns elements to the output array based on hash
    }

    bishopMagic.push_back(magic);
  }
  
  //everything below is just outputting the values into arrays in a seperate file for reuse

  file << "static const uint64_t bishopMagicNums[64] = {\n  ";
  for(int i = 0; i < 64; i++){
    file << bishopMagic[i];
    if(i + 1 < 64) file  << "ULL, ";
  }

  file << "ULL\n};\n";

  std::ofstream attackFile("attackMasks.hpp");
  std::ofstream bitFile("bitOffset.hpp");
  attackFile << "#include <cstdint>\n\nstatic const uint64_t rookAttackTable[64][4096] = {\n";
  bitFile << "#include <cstdint>\n\nstatic const uint64_t rookBitTable[64] = {";
  for(int pos = 0; pos < 64; pos++){
    bitFile << 64 - rookBits[pos];
    if(pos + 1 < 64) bitFile << ", ";
    attackFile << " {";
    for(int i = 0; i < rookAttackArray[pos].size(); i++){
      attackFile << rookAttackArray[pos][i] << "ULL";
      if(i + 1 < rookAttackArray[pos].size()) attackFile << ", ";
    }
    attackFile << " },\n";
  }
  bitFile << "\n};\n";
  attackFile << "};\n";

  attackFile << "static const uint64_t bishopAttackTable[64][512] = {\n";
  bitFile << "static const uint64_t bishopBitTable[64] = {";
  for(int pos = 0; pos < 64; pos++){
    bitFile << 64 - bishopBits[pos];
    if(pos + 1 < 64) bitFile << ", ";
    attackFile << " {";
    for(int i = 0; i < bishopAttackArray[pos].size(); i++){
      attackFile << bishopAttackArray[pos][i] << "ULL";
      if(i + 1 < bishopAttackArray[pos].size()) attackFile << ", ";
    }
    attackFile << " },\n";
  }
  bitFile << "\n};\n";
  attackFile << "};\n";

  std::array<uint64_t, 64> piece;
  piece = kingMask();
  attackFile << "static const uint64_t kingAttackTable[64] = {\n  ";
  for(int i = 0; i < 64; i ++){
    attackFile << piece[i] << "ULL";
    if(i + 1 < 64) attackFile <<", ";
  }
  attackFile << "\n};\n";
  std::array<uint64_t, 64> knightPiece;
  knightPiece = knightMask();
  attackFile << "static const uint64_t knightAttackTable[64] = {\n";
  for(int i = 0; i < 64; i ++){
    attackFile << knightPiece[i] << "ULL";
    if(i + 1 < 64) attackFile <<", ";
  }
  attackFile << "\n};\n";

  std::array<uint64_t, 128> pawn;
  pawn = pawnMask();
  attackFile << "static const uint64_t pawnAttackTable[128] = {\n";
  for(int i = 0; i < 128; i ++){
    attackFile << pawn[i] << "ULL";
    if(i + 1 < 128) attackFile <<", ";
  }
  attackFile << "\n};\n";

  attackFile << "static const uint64_t rookMaskTable[64] = {\n";
  for(int i = 0; i < 64; i ++){
    attackFile << rookMaskArray[i] << "ULL";
    if(i + 1 < 64) attackFile <<", ";
  }
  attackFile << "\n};\n";

  attackFile << "static const uint64_t bishopMaskTable[64] = {\n";
  for(int i = 0; i < 64; i ++){
    attackFile << bishopMaskArray[i] << "ULL";
    if(i + 1 < 64) attackFile <<", ";
  }
  attackFile << "\n};\n";

  attackFile.close();
  file.close();
}