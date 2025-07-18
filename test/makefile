CXXFLAGS = -std=c++11
NVCCFLAGS = -arch=sm_89 -g -G
LIBS = -lcudnn -lcublas -lcudart -lcurand
SRCDIR = ./source
HDRDIR = ./header
CUSRC = $(wildcard $(SRCDIR)/*.cu)
CUOBJ = $(notdir $(CUSRC:.cu=.o))



.PHONY : program clean

program: main
	rm -rf *.o

main : main.o $(CUOBJ)
	g++ $(CXXFLAGS) -o $@ main.o $(CUOBJ) $(LIBS)
	./main

main.o : main.cpp
	g++ $(CXXFLAGS) -c $< -o $@

%.o : $(SRCDIR)/%.cu
	nvcc $(NVCCFLAGS) -I$(HDRDIR) -c $< -o $@



arch : arch.cu
	nvcc arch.cu -o arch
	./arch

clean:
	rm -rf arch *.o main