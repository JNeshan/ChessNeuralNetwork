//transpositionTable.h

#ifndef TRANSPOSITIONTABLE_H
#define TRANSPOSITIONTABLE_H

#include <vector>
#include <cstdint>
#include <cstddef>


extern int mateScore;

enum tableFlag : uint8_t{NONE, EXACT, LOWER, UPPER};

struct tableEntry {
  int depth;
  int score;
  int rtDist;
  uint64_t storedKey;
  uint16_t bestMove;
  tableFlag flag;
  tableEntry();
};

class transpositionTable {
public:
  std::vector<tableEntry> table;
  size_t entries;
  size_t mask;

  transpositionTable(size_t s); //initializer
  void store(uint64_t key, int depth, int16_t score, tableFlag flag, uint16_t bestMove, int rootDist); //used to store new entries in table
  tableEntry* getTable(uint64_t potentialKey); //retrieves table
};

#endif