//chessState.cpp

#include <iostream>
#include <fstream>
#include "../header/chessState.h"
#include<exception>
std::ofstream output("simGame.txt");

inline int position(int rank, int file){
  return rank * 8 + file;
}

inline std::string toAlgebraic(int pos){
  return std::string(1, 'a' + pos%8) + std::to_string(1 + pos/8);
}

inline uint16_t fromAlgebraic(std::string move){
  char fFile = move[0], fRank = move[1]; //seetup variables for conversion
  char tFile = move[2], tRank = move[3];
  int from = (fRank - '1') * 8 + (fFile - 'a');
  int to = (tRank - '1') * 8 + (tFile - 'a');

  uint16_t packedMove = (from << 6) | to;

  if(move.length() == 5){ //promotion handling
    switch (tolower(move[4])){
      case 'n': packedMove |= (1 << 13); break;
      case 'b': packedMove |= (1 << 12); break;
      case 'q': packedMove |= (1 << 14); break;
      case 'r': packedMove |= (1 << 15); break;
      default: break;
    }
  }

  return packedMove;

}

const std::array<const uint64_t, 4> chessState::castleSpaces = { //used to verify spaces are empty for castling
  (1ULL << 5) | (1ULL << 6),
  (1ULL << 1) | (1ULL << 2) | (1ULL << 3),
  (1ULL << 61) | (1ULL << 62),
  (1ULL << 57) | (1ULL << 58) | (1ULL << 59)
};
const std::array<const int, 4> chessState::castleStateChecks = {0b0001, 0b0010, 0b0100, 0b1000};
const std::array<PieceType, 6> chessState::allPieces  = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}; 
const std::array<Color, 2> pieceColors = {WHITE, BLACK};
const std::array<int, 6> chessState::pieceValues = {100, 300, 330, 500, 900, 20000}; //change
int chessState::nodesExplored = 0;

chessState::chessState(std::string iState){  
  tTable = new transpositionTable(512);
  int rank = 0, file = 0; int i = 0;
  while(true){
    char c = iState[i];
    if(c == '/'){
      iState.erase(0, i + 1);
      file = 0; i = 0; rank++;
      continue;
    }

    if(c == ' '){
      iState.erase(0, i + 1);
      break;
    }

    if(isdigit(c)){
      file += c - '0';
      i++;
      continue;
    }
    Color color = isupper(c) ? WHITE : BLACK;
    int pos = (7 - rank) * 8 + file;
    uint64_t bit = 1ULL << pos;
    occupied[color] |= bit;
    
    char pieceChar = tolower(c);

    PieceType type;
    switch(pieceChar) {
      case 'p': type = PAWN; break;
      case 'n': type = KNIGHT; break;
      case 'r': type = ROOK; break;
      case 'b': type = BISHOP; break;
      case 'q': type = QUEEN; break;
      case 'k': type = KING; break;
    }

    bitboards[color][type] |= bit; //specifies where each piece of a certain color and type are located
    occupied[color] |= bit; //assigns each players piece locations to their own board
    i++;
    file++;
  }

  std::istringstream stream(iState);
  std::string special;

  stream >> special; //set player turn
  active = (special == "w") ? WHITE : BLACK;

  stream >> special; //set stil lvalid castling
  for (char c : special) {
    switch (c) {
      case 'K': castleState |= 0b0001; break;
      case 'Q': castleState |= 0b0010; break;
      case 'k': castleState |= 0b0100; break;
      case 'q': castleState |= 0b1000; break;
      default: break;
    }
  }
  
  stream >> special; //set enpassant space
  if(special == "-"){
    enpassant = 0ULL;
  }
  else{
    int pos = (special[0] - 'a') + ((special[1] - '1') * 8);
    enpassant |= (1ULL << pos);
  }
  
  stream >> special; //set half turn clock
  halfTurns = std::stoi(special);
  
  stream >> special; //set full turn count
  fullTurns = std::stoi(special);

  zobStruct = new zobristKeys();
  initializeZobristKey();
}

chessState::chessState(const chessState& right){
  this->active = right.active;
  this->bitboards = right.bitboards;
  this->occupied = right.occupied;
  this->enpassant = right.enpassant;
  this->castleState = right.castleState;
  this->fullTurns = right.fullTurns;
  this->halfTurns = right.halfTurns;
  this->lMoves = right.lMoves;
  this->rMoves = right.rMoves;
  this->zobStruct = right.zobStruct;
  this->currentKey = right.currentKey;
  this->tTable = right.tTable;
}

uint64_t chessState::retrieveAttackBoard(int pos, Color player, PieceType piece, uint64_t state){
  switch (piece)
  {
  case ROOK:{
    uint64_t blockers = rookMaskTable[pos] & (state), magic = rookMagicNums[pos], offset = rookBitTable[pos];
    magic = ((blockers * magic) >> offset);
    return rookAttackTable[pos][magic];
  }
  case BISHOP:{
    uint64_t blockers = bishopMaskTable[pos] & (state), magic = bishopMagicNums[pos], offset = bishopBitTable[pos];
    magic = ((blockers * magic) >> offset);
    return bishopAttackTable[pos][magic];
  }
  case QUEEN:
  {
    return retrieveAttackBoard(pos, player, ROOK, state) | retrieveAttackBoard(pos, player, BISHOP, state);
  }
  case KING:{
    return kingAttackTable[pos];
  }
  case KNIGHT:{
    return knightAttackTable[pos];
  }
  case PAWN:{
    return pawnAttackTable[pos + (64 * player)];
  }
  default:
    return 0;
  }
}

void chessState::printBoard(){
  for(int i = 7; i >= 0; i--){
    output<<"  -------------------------------------\n"<<i+1<<" ";
    for(int j = 0; j < 8; j++){
      output<<sPieceAt(8*i + j)<<" | ";
    }
    output<<std::endl;
  }
  output<<"  -------------------------------------\n";
  output<<"  a    b    c    d    e    f    g    h"<<std::endl;
  if(!active){
    output<<"White to move"<<std::endl;
  }
  else{
    output<<"Black to move"<<std::endl;
  }
}

char pieceChars[2][6] = { //printing
  {'P', 'N', 'B', 'R', 'Q', 'K'},
  {'p', 'n', 'b', 'r', 'q', 'k'}
};

std::string sPieceChars[2][6] = {
  {"♟", "♞", "♝", "♜", "♛", "♚"},
  {"♙", "♘", "♗", "♖", "♕", "♔"}
};

std::string chessState::sPieceAt(int pos){
  uint64_t bitPos = 1ULL << pos;
  for(int color = 0; color < 2; color++){
    for(int type = 0; type < 6; type++){
      if(bitboards[color][type] & bitPos){
        return sPieceChars[color][type];
      }
    }
  }
  return "　";
}

char chessState::pieceAt(int pos){
  uint64_t bitPos = 1ULL << pos;
  for(int color = 0; color < 2; color++){
    for(int type = 0; type < 6; type++){
      if(bitboards[color][type] & bitPos){
        return pieceChars[color][type];
      }
    }
  }
  return '.';
}

bool chessState::isThreatenedBit(int pos){
  for(auto p : allPieces){
    uint64_t attack = retrieveAttackBoard(pos, static_cast<Color>(active), p, (occupied[0] | occupied[1]));
    uint64_t enemyPieces = bitboards[!active][p] & occupied[!active];
    if(attack & enemyPieces) return true;
  }
  return false;
}

//uses inversed attack logic to detect if a space is threatened
bool chessState::isThreatenedBit(int pos, uint64_t state){
  for(auto p : allPieces){
    uint64_t attack = retrieveAttackBoard(pos, static_cast<Color>(active), p, state);
    uint64_t enemyPieces = (bitboards[!active][p] & occupied[!active]); //
    if(attack & enemyPieces) return true;
  }
  return false;
}
//unused
bool chessState::isThreatenedBit(int pos, uint64_t state, Color player){

  for(auto p : allPieces){
    uint64_t attack = retrieveAttackBoard(pos, player, p, state);
    uint64_t enemyPieces = (bitboards[!player][p] & occupied[!player]);
    if(attack & enemyPieces) return true;
  }
  return false;
}

std::vector<uint16_t> chessState::getAllMovesBit(){
  std::srand(std::time(0));
  std::vector<uint16_t> moves(0);
  std::vector<uint16_t> store(0);
  //creates move vectors, move logic calls
  store = pawnMoves(bitboards[active][PAWN]);
  moves.insert(moves.begin(), store.begin(), store.end());
  store = slidingMoves(bitboards[active][ROOK], ROOK);
  moves.insert(moves.begin(), store.begin(), store.end());
  store = slidingMoves(bitboards[active][BISHOP], BISHOP);
  moves.insert(moves.begin(), store.begin(), store.end());
  store = singleMoves(bitboards[active][QUEEN], QUEEN);
  moves.insert(moves.begin(), store.begin(), store.end());
  store = singleMoves(bitboards[active][KNIGHT], KNIGHT);
  moves.insert(moves.begin(), store.begin(), store.end());
  store = singleMoves(bitboards[active][KING], KING);
  moves.insert(moves.begin(), store.begin(), store.end());  

  return moves;
}

std::vector<uint16_t> chessState::pawnMoves(uint64_t mappings){
  std::vector<uint16_t> moves;
  while(mappings){
    int pos = __builtin_ctzll(mappings);
    uint64_t available = retrieveAttackBoard(pos, active, PAWN, (occupied[0] | occupied[1])); //requires enemy pieces to capture
    available &= (occupied[!active] | enpassant);
    while(available){
      int newPos = __builtin_ctzll(available);
      uint16_t packedMove = (pos << 6) | newPos; //packing the two ints into a binary
      
      if((newPos) / 8 == 0 || (newPos) / 8 == 7){ //adds additional flag if promotion occurs
        moves.push_back(packedMove | (1<<12)); //bishop
        moves.push_back(packedMove | (1<<13)); //knight
        moves.push_back(packedMove | (1<<14)); //queen
        moves.push_back(packedMove | (1<<15)); //rook
      }
      else{
        moves.push_back(packedMove); 
      }
      available ^= (1ULL << newPos);
    }

    int dir = !active ? 8 : -8; //white is 0 so inverse means go up 8
    available = (1ULL << (pos + dir)); //sets piece to forward move rank

    if(available & ~(occupied[0] | occupied[1])){ //checks if single move space is unoccupied
      uint16_t packedMove = (pos << 6) | pos+dir; //packing the two ints into a binary
      moves.push_back(packedMove);
      available = (1ULL << (pos + 2*dir)); //sets piece to double forward position preemptively

      if((pos/8) - (5 * active) == 1 && available & ~(occupied[0] | occupied[1])){ //checks if double move can be made
        uint16_t packedMove = (pos << 6) | pos+(2*dir); //packing the two ints into a binary
        moves.push_back(packedMove); 
      }
      else if((pos + dir) / 8 == 0 || (pos + dir) / 8 == 7){ //promotion check, can't occur on double move
        moves[moves.size() - 1] = (packedMove | (1<<12)); //bishop, replaces the unflagged entry here
        moves.push_back(packedMove | (1<<13)); //knight
        moves.push_back(packedMove | (1<<14)); //queen
        moves.push_back(packedMove | (1<<15)); //rook
      }
    }
    mappings ^= (1ULL << pos);
  }
  return moves;
}

std::vector<uint16_t> chessState::slidingMoves(uint64_t mappings, PieceType piece){
  std::vector<uint16_t> moves;
  while(mappings){
    int pos = __builtin_ctzll(mappings);
    uint64_t available = retrieveAttackBoard(pos, active, piece, (occupied[0] | occupied[1])) & ~occupied[active]; //gives the positions piece can move to if an allied piece isn't there
    while(available){
      int newPos = __builtin_ctzll(available);
      uint16_t packedMove = (pos << 6) | newPos; //packing the two ints into a binary
      moves.push_back(packedMove); 
      available ^= (1ULL << newPos);
    }
    mappings ^= (1ULL << pos);
  }
  return moves;
}

std::vector<uint16_t> chessState::singleMoves(uint64_t mappings, PieceType piece){
  std::vector<uint16_t> moves;
  while(mappings){
    int pos = __builtin_ctzll(mappings);
    uint64_t available = retrieveAttackBoard(pos, active, piece, (occupied[0] | occupied[1])) & ~(occupied[active]); //gives the positions piece can move to if an allied piece isn't there
    while(available){
      int newPos = __builtin_ctzll(available);
      uint16_t packedMove = (pos << 6) | newPos;
      moves.push_back(packedMove); 
      available ^= (1ULL << newPos);
    }

    if(piece == KING){
      if(castlingCheck(pos, false)){ //kingside
        uint16_t packedMove = (pos << 6) | pos + 2;
        moves.push_back(packedMove);
      }
      if(castlingCheck(pos, true)){ //queenside
        uint16_t packedMove = (pos << 6) | pos - 2;
        moves.push_back(packedMove);
      }
    }
    mappings ^= (1ULL << pos);
  }
  return moves;
}

bool chessState::castlingCheck(int pos, bool side){
  if(!(castleState & castleStateChecks[2*active+side])){ //if relevant piece has moved castling disallowed
    return false; 
  } 
  if(castleSpaces[2*active+side] & (occupied[active] | occupied[!active])){
    return false;
  }
  return true;
}

bool chessState::legalMove(uint16_t move){ //checks if moves are legal, creates temporary state values without mutating members
  int initial = (move >> 6) & 0x3F, destination = move &0x3F;
  uint64_t newState = (((occupied[0] | occupied[1] ) & ~(1ULL << initial)) | (1ULL << destination));

  int kingPos = __builtin_ctzll(bitboards[active][KING]);
  if(destination == kingPos){
    std::cout<<"King capture passed"<<std::endl;
    printBoard();
    throw std::runtime_error("king capture");
    return false;
  }

  if(bitboards[active][KING] & (1ULL << initial)){
    if(destination - initial == 2){ //castling check safety, other checks handeled in pseudo-valid
      if(isThreatenedBit(initial) || isThreatenedBit(initial+1) || isThreatenedBit(initial+2)){
        return false;
      }
      else{
        return true;
      }
    }
    else if(destination - initial == -2){
      if(isThreatenedBit(initial) || isThreatenedBit(initial-1) || isThreatenedBit(initial-2)){
        return false;
      }
      else{
        return true;
      }
    }
    int kingPos = __builtin_ctzll(bitboards[active][KING]);
    uint64_t hold = occupied[!active];
    occupied[!active] &= ~(1ULL << destination);
    bool legal = !isThreatenedBit(destination, newState);
    occupied[!active] = hold;
    return legal;
  }
  else{
    kingPos = __builtin_ctzll(bitboards[active][KING]);
    uint64_t hold = occupied[!active];
    occupied[!active] &= ~(1ULL << destination);

    bool legal = !isThreatenedBit(kingPos, newState);
    occupied[!active] = hold;    
    return legal;
  }
  
}

bool chessState::updateBoard(uint16_t move){ //handles toggling the zobrist key
  
  nodesExplored++;

  uint64_t tmpLeft = lMoves >> 16, tmpRight = rMoves >> 16; //update previous moves
  tmpLeft |= (rMoves &0xFFFFULL) << 48;
  tmpRight |= (uint64_t(move) << 48);
  lMoves = tmpLeft; rMoves = tmpRight;
  if((lMoves == rMoves) && lMoves){
    return true;
  }

  uint64_t initial = 1ULL << ((move >> 6) & 0x3F), final = 1ULL << (move & 0x3F); //unpacks move to bitboard
  int nPos = move & 0x3F, iPos = (move >> 6) & 0x3F; //values for updating zobrist key
  currentKey ^= zobStruct->blackToMove; //always toggles for turn switch
  if(enpassant){
    int enPos = __builtin_ctzll(enpassant);
    currentKey ^= zobStruct->enpassantFile[enPos%8]; //toggles enpassant
  }

  for(auto p : allPieces){
    if(bitboards[active][p] & initial){ //finds the piece being moved
      currentKey ^= zobStruct->pieceSquare[p + (6 * active)][iPos]; //toggles old piece position
      if(p == PAWN){
        halfTurns = 0;
        if(final == enpassant){
          bitboards[!active][p] &= ~(active ? final << 8: final >> 8); //removes captured pawn
          occupied[!active] &= ~(active ? final << 8: final >> 8);
          enpassant = 0ULL; //no new enpassant
          currentKey ^= zobStruct->pieceSquare[PAWN + (6 * !active)][nPos + (8*!active) - (8*active)]; //toggle enpassanted pawn
        }
        else if(abs((move & 0x3F) - ((move >> 6) & 0x3F)) == 16){
          enpassant = active ? final << 8: final >> 8; //sets new enpassant position
          currentKey ^= zobStruct->enpassantFile[nPos%8];
        }
      }
      else{
        if(occupied[0] & occupied[1]){
          halfTurns = 0;
        }
        else{
          halfTurns++; //no pawn advance or capture
        }
        
        enpassant = 0ULL;
      }
      switch (move >> 12 & 0xF) //promotion flags, preemptively removes old pawn position then sets p to new piece for updates
      {
      case 0b0001:{
        bitboards[active][p] ^= initial;
        p = BISHOP;
        break;
      }
      case 0b0010:{
        bitboards[active][p] ^= initial;
        p = KNIGHT;
        break;
      }
      case 0b0100:{
        bitboards[active][p] ^= initial;
        p = QUEEN;
        break;
      }
      case 0b1000:{
        bitboards[active][p] ^= initial;
        p = ROOK;
        break;
      }
      default:{
        if(p == KING){
          int dis = nPos - iPos;
          if(dis == 2){ //kingside
            int rkIPos = iPos + 3, rkNPos = nPos - 1;
            bitboards[active][ROOK] &= ~(1ULL << rkIPos); //removes rook from old position
            bitboards[active][ROOK] |= (1ULL << rkNPos); //adds rook to new position
            castleState &= ~castleStateChecks[2*active];
            occupied[active] &= ~(1ULL << rkIPos);
            occupied[active] |= (1ULL << rkNPos);
            
            currentKey ^= zobStruct->pieceSquare[ROOK + (6 * active)][rkIPos];
            currentKey ^= zobStruct->pieceSquare[ROOK + (6 * active)][rkNPos];
            currentKey ^= zobStruct->castlingRights[2*active];            
          }
          else if(dis == -2){ //queenside
            int rkIPos = iPos - 4, rkNPos = nPos + 1;
            bitboards[active][ROOK] &= ~(1ULL << rkIPos);
            bitboards[active][ROOK] |= (1ULL << rkNPos);
            occupied[active] &= ~(1ULL << rkIPos);
            occupied[active] |= (1ULL << rkNPos);
            castleState &= ~castleStateChecks[2*active+1];
            
            currentKey ^= zobStruct->pieceSquare[ROOK + (6 * active)][rkIPos];
            currentKey ^= zobStruct->pieceSquare[ROOK + (6 * active)][rkNPos];
            currentKey ^= zobStruct->castlingRights[1 + (2*active)];
          }
          if(castleState & castleStateChecks[2*active+1]){
            castleState &= ~castleStateChecks[2*active+1];
            currentKey ^= zobStruct->castlingRights[1 + (2*active)];
          }
          if(castleState & castleStateChecks[2*active]){
            currentKey ^= zobStruct->castlingRights[2*active];
            castleState &= ~castleStateChecks[2*active];
          }
        }
        else if(castleState & castleStateChecks[2*active+1] && initial == 1ULL << (56*active)){ //queenside rook moves
          castleState ^= castleStateChecks[2*active+1]; //toggles castle state and zobrist key 
          currentKey ^= zobStruct->castlingRights[1 + (2 * active)];

        }
        else if(castleState & castleStateChecks[2*active] && initial == 1ULL << (7 + 56*active)){ //kingside rook moves
          castleState ^= castleStateChecks[2*active]; //toggles castle state and zobrist key 
          currentKey ^= zobStruct->castlingRights[2 * active];
        }
        
        bitboards[active][p] ^= initial; //removes old position
        break;
      }
      }


      currentKey ^= zobStruct->pieceSquare[p + (6 * active)][nPos]; //toggles new piece position
      bitboards[active][p] |= 1ULL << (move & 0x3F);
      occupied[active] ^= initial; //removes old position from full board
      occupied[active] |= 1ULL << (move & 0x3F); //adds new position to players full board

      occupied[static_cast<Color>(!active)] &= ~occupied[active]; //removes captured piece from full board
      fullTurns += active; //increments on black move
      active = static_cast<Color>(!active); //changes player
      for(auto e: allPieces){
        if((final & bitboards[active][e])){ //checks if one of piece type is on the moved to space
          currentKey ^= zobStruct->pieceSquare[e + (6 * active)][nPos];
          bitboards[active][e] &= ~occupied[static_cast<Color>(!active)]; //removes captured piece from piece type, ensures no overlaps
          if(e == ROOK){ //edge case incase an unmoved rook is captured with its castle available
            if(nPos == 7 + 56*active && castleState & castleStateChecks[2*active]){ //checks if an unmoved rook is captured kingside
              currentKey ^= zobStruct->castlingRights[2*active]; //toggle castle rights
              castleState &= ~castleStateChecks[2*active];
            }
            else if(nPos == 56*active && castleState & castleStateChecks[1 + 2*active]){ //checks if an unmoved rook is captured queenside
              currentKey ^= zobStruct->castlingRights[2*active + 1]; //toggle castle rights
              castleState &= ~castleStateChecks[2*active + 1];
            }
          }
          break;
        }
      }
      if(halfTurns == 50){
        return true;
      }
      return false;
    }
  }
  return false;
}

void chessState::playerMove(std::string move){
  output<<move<<"\n\n";
  uint16_t m = fromAlgebraic(move);
  updateBoard(m);
}

void chessState::initializeZobristKey(){
  currentKey = 0ULL;
  for(int pos = 0; pos < 64; pos++){
    std::pair<PieceType, Color> posInfo = pieceAtZob(pos);
    if(posInfo.first != NOPIECE){
      int zobristIndex = zobStruct->pieceSquare[posInfo.first + (6 * posInfo.second)][pos];
      currentKey ^= zobStruct->pieceSquare[posInfo.first + (6 * posInfo.second)][pos];
    }
  }

  if(active == BLACK)
    currentKey ^=  zobStruct->blackToMove;

  if(castleState & castleStateChecks[0]) currentKey ^= zobStruct->castlingRights[0];
  if(castleState & castleStateChecks[1]) currentKey ^= zobStruct->castlingRights[1];
  if(castleState & castleStateChecks[2]) currentKey ^= zobStruct->castlingRights[2];
  if(castleState & castleStateChecks[3]) currentKey ^= zobStruct->castlingRights[3];

  if(enpassant != 0ULL){
    int file = __builtin_ctzll(enpassant)%8;
    currentKey ^= zobStruct->enpassantFile[file];
  }
}

std::pair<PieceType, Color> chessState::pieceAtZob(int pos){
  uint64_t pieceBit = 1ULL << pos;
  if(pieceBit & ~(occupied[0] | occupied[1])){
    return std::make_pair(NOPIECE, NOCOLOR);
  }
  for(int c = 0; c < 2; c++){
    Color color = static_cast<Color>(c);
    for(int p = 0; p < 6; p++){
      PieceType piece = static_cast<PieceType>(p);
      if(bitboards[c][p] & pieceBit){
        return std::make_pair(piece, color);
      }
    }
  }
  return std::make_pair(NOPIECE, NOCOLOR);
}

uint64_t chessState::getKey(){
  return currentKey;
}