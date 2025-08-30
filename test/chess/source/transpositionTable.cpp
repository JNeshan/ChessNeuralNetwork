//transpositionTable.cpp

#include <vector>
#include <cstdint>
#include <cstddef>
#include "../header/transpositionTable.h"

int mateScore = 1000000;
tableEntry::tableEntry() : parent(this), children(), n(0){
  p = 0; //run through neural network here or later
  w = p;
}
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

void transpositionTable::store(uint64_t key, tableEntry* par){
  size_t index = key & mask;
  tableEntry& entry = table[index];
  if(entry.n == 0){
    entry.n = 1;
    entry.parent = par;
    entry.w = 0;
    entry.p = 0; //use neural network here or somewhere else
  }
}

void transpositionTable::backPropagate(tableEntry* node){
  tableEntry* ptr = node;
  while(ptr != ptr->parent){
    ptr = ptr->parent;
    ptr->w += node->p;
  }
}

tableEntry* transpositionTable::getTable(uint64_t potentialKey){
  int index = potentialKey & mask; //gets indexed based off hash and mask indexing
  return &table[index];
}


