# Alpha Zero Based Neural Network for Self Teaching Chess

## Project Description
This project is a recreation of the structure of the alpha-zero neural network and Monte-Carlo Tree Search algorithm used to create a high level self teaching chess bot. The chess library is handeled through a custom framework I created previously that uses bitboards to model the chess states. The chess frameworks only major modification is the removal of its own heuristic derived AI players and making it give tensorization private member access so it can convert a given states bitboard into an appropriate initial tensor to use with the network. 


## Classes
### Layer
### NeuralNetwork
###





## Relevant Information
### AlphaZero
AlphaZero is a residual neural network that is a more generalized version of the AlphaGo Zero algorithm. AlphaZero was made to compete at a few games including chess. It's signifigance was due to being capable of outperforming stockfish, the best modern chess engine, despite using a generalized neural network.

### Bitboards
A bitboard is an x bit integer represented by its binaray digits. A chess board has 64 spaces and is thus represented as a 64 bit integer where a 1 can represent an occupied space and a 0 unoccupied. The bitboard framework uses 64 bit bitboards for each piece type for each player. Moves and board updates utilize the bitboards so that updates only require simple bit switching operations making them the fastest method (the used framework was able to generate an average of 2 million moves a second with a heuristic search)

### Monte-Carlo Tree Search
