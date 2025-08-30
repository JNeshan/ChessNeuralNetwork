//transpositionTable.cpp

#include <vector>
#include <cstdint>
#include <cstddef>
#include "../header/transpositionTable.h"

int mateScore = 1000000;
tableEntry::tableEntry() : depth(0), score(0), rtDist(0), storedKey(0ULL), bestMove(0ULL) {}
//s is used to create a large transpose table, this on top of the large amount of keys generated causes a pretty meaty compile time
transpositionTable::transpositionTable(size_t s){
  entries = (s * 1024 * 1024) / sizeof(tableEntry);

  size_t bits = 1;
  while(bits * 2 <= entries){ 
    bits *= 2;
  }
  entries = bits; //ensures a proper power of 2
  mask = entries - 1; //mask is essentially a bit representation of the number of elements, performing -1 means that all bits behind the MSB are 1 which makes for easy indicing
  table.resize(entries);
}

void transpositionTable::store(uint64_t key, int depth, int16_t score, tableFlag flag, uint16_t bestMove, int rootDist){
  
  size_t index = key & mask;
  tableEntry& entry = table[index];
  //replaces same index entry if its unique and newer or has a higher depth
  if(entry.flag == NONE || entry.storedKey != key || depth >= entry.depth){
    entry.storedKey = key;
    entry.depth = depth;
    entry.score = score;
    entry.flag = flag;
    entry.bestMove = bestMove;
    entry.rtDist = rootDist;
  }
}

tableEntry* transpositionTable::getTable(uint64_t potentialKey){
  int index = potentialKey & mask; //gets indexed based off hash and mask indexing
  if(table[index].storedKey == potentialKey){ //verifies the key stored at index is identical to the one being searched for
    return &table[index];
  }
  return nullptr;
}


