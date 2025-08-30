//chessState.h
#ifndef CHESSSTATE_H
#define CHESSSTATE_H

#include <vector>
#include <string>
#include <array>
#include <map>
#include <cstdint>
#include <utility>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include "attackMasks.hpp"  //files contain precomputed tables that are vital for this program
#include "magicNumbers.hpp" //the mask tables and magic hashing values are computed in generateTables (change name)
#include "bitOffset.hpp"    //it does not need to be rerun them but its left to show that I computed all the data manually, but it will still work perfectly if it is
#include "zobristKeys.h"
#include "transpositionTable.h"


//enum types


class chessState{
public:

  //testing function for manual opponent

  

  int halfTurns; //half turn clock
  int fullTurns; //full turn clock
  int maxNodes = 5000000;

  zobristKeys* zobStruct; //used to create the key for indexing transpose table

  //constructors
  chessState(std::string iState); //forsyth-edwards based constructor
  chessState(const chessState& right); //copy constructor, unimplemented

  transpositionTable* tTable; //holds the found transpose tables, indexed using hash key

  std::map<std::string, int> blackScores;
  std::map<std::string, int> whiteScores;

  
  //the real functions
  void printBoard(); //outputs current board state for testing
  char pieceAt(int pos);
  std::pair<PieceType, Color> pieceAtZob(int pos);
  std::vector<uint16_t> getAllMovesBit();
  std::string searchMove(); //starts minimax search and returns the best move
  void playerMove(std::string move);
  int minimaxSearch(chessState& state, int depth, bool maxer);
  int evaluationHeuristic(chessState& state);
  int minimaxSearchAB(chessState& state, int depth, int a, int b, int dist, int& mNodes);
  std::string sPieceAt(int pos);

  void initializeZobristKey(); //generates zobrist key from board state, used only at initialization
  
  bool updateBoard(uint16_t move); //updates board state to reflect move
  //additionally handles the XOR operations required to update the zobrist key that indexes the zobrist table
  //operations occur at the same time the actual relevant board update operations occur for the most part
  
  //attacks
  bool isThreatenedBit(int pos); //check for current state
  bool isThreatenedBit(int pos, uint64_t state); //check for different state
  bool isThreatenedBit(int pos, uint64_t state, Color player);
  friend struct zobristKeys;
  private:

  
  //game state variables
  std::array<std::array<uint64_t, 6>, 2> bitboards{}; //bit position of each piece of each type a player controls
  Color active; //denotes which player is moving
  uint64_t enpassant = 0ULL; //marks enpassant position bit
  uint64_t fullBoard; //bit position of every piece
  uint64_t lMoves, rMoves; //track the last 8 moves
  uint64_t currentKey; //transpose table functionality
  std::array<uint64_t, 2> occupied{}; //bit positions of each piece a player controls 
  int castleState = 0b0000; //represents which castle moves fulfill the unmoved condition; wKside, wQside, bKside, bQside  

  //reused constants
  static const std::array<PieceType, 6> allPieces;  //references piece types for enum
  static const std::array<const int, 4> castleStateChecks; //references castle state bits
  static const std::array<const uint64_t, 4> castleSpaces; //references positions relevant to castling
  static const std::array<int, 6> pieceValues; //stores the value of a piece type for the heuristic
  static int nodesExplored;


  //move generation
  std::vector<uint16_t> pawnMoves(uint64_t pos); //move generation for pawn
  std::vector<uint16_t> slidingMoves(uint64_t pos, PieceType piece); //move generation for rook, bishop, and queen
  std::vector<uint16_t> singleMoves(uint64_t pos, PieceType piece); //move generation for king and knight
  bool castlingCheck(int pos, bool side); //checks if castling spaces are unoccupied and castling pieces haven't moved
  uint64_t retrieveAttackBoard(int pos, Color player, PieceType piece, uint64_t state); //gets the relevant bitboard representing the spaces a piece can move to attack
  bool legalMove(uint16_t move); //checks if a move doesn't result in a check
  
  

  // #endregion
};




#endif