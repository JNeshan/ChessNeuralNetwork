//transpositionTable.h

#ifndef TRANSPOSITIONTABLE_H
#define TRANSPOSITIONTABLE_H

#include <vector>
#include <cstdint>
#include <cstddef>

struct tableEntry {
  tableEntry* parent;
  std::vector<tableEntry*> children;
  int n;
  double w, p;
  tableEntry();
};

class transpositionTable {
public:
  std::vector<tableEntry> table;
  size_t entries;
  size_t mask;

  transpositionTable(size_t s); //initializer
  void store(uint64_t key, tableEntry* par); //used to store new entries in table
  void backPropagate(tableEntry* node);
  tableEntry* getTable(uint64_t potentialKey); //retrieves table
};

#endif