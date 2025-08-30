CXXFLAGS = -std=c++11
NVCCFLAGS = -arch=sm_89 -g -G
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


.PHONY : program clean

program: main
	rm -rf *.o

main : main.o $(CHESSOBJ) $(CUOBJ)
	g++ $(CXXFLAGS) -o $@ main.o $(CHESSOBJ) $(CUOBJ) $(LIBS)
	./main

main.o : main.cpp
	g++ $(CXXFLAGS) -c $< -o $@

%.o : $(SRCDIR)/%.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o : $(CHESSSDIR)/%.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $< -o $@

arch : arch.cu
	nvcc arch.cu -o arch
	./arch

clean:
	rm -rf arch *.o main