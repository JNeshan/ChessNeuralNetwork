CXXFLAGS = -std=c++17 -O2
NVCCFLAGS = -arch=sm_89 -g -G -O2
LIBS = -lcudnn -lcublas -lcudart -lcurand
SRCDIR = ./source
HDRDIR = ./header
CHESSSDIR = ./chess/source
CHESSHDIR = ./chess/header
CHESSSRC = $(wildcard $(CHESSSDIR)/*.cpp)
CHESSOBJ = $(notdir $(CHESSSRC:.cpp=.o))
CUSRC = $(wildcard $(SRCDIR)/*.cu)
CUOBJ = $(notdir $(CUSRC:.cu=.o))
INCLUDES = -I$(HDRDIR) -I$(CHESSHDIR)
CUDA_INC = -I/usr/local/cuda-12.9/include/


.PHONY : program clean
.PRECIOUS : main
program: main

debug: main
	./clrChess.sh
	gdb ./main


main : main.o $(CHESSOBJ) $(CUOBJ)
	g++ $(CXXFLAGS) $(CUDA_INC) -o $@ main.o $(CHESSOBJ) $(CUOBJ) $(LIBS)
	./clrChess.sh
	./main

main.o : main.cpp
	g++ $(CXXFLAGS) $(CUDA_INC) $ -c $< -o $@

%.o : $(SRCDIR)/%.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o : $(CHESSSDIR)/%.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $< -o $@

arch : arch.cu
	nvcc arch.cu -o arch
	./arch

clean:
	rm -rf arch *.o main