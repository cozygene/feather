CXX=clang++

# Change paths to boost and Eigen installations
CXXFLAGS=-std=c++11 -Wall -pedantic -O3 -DNDEBUG -pthread -I/usr/local/Cellar/boost/1.67.0_1/include/ -I/usr/local/Cellar/eigen/3.3.5/include/eigen3/

LDFLAGS=-O3 -pthread  -lboost_filesystem -lboost_system -lboost_thread-mt -lboost_program_options 

BIN=permutation_testing_samc

SRC=$(wildcard permutation_testing_samc.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $(LDFLAGS) -o $(BIN) $^

%.o: %.c
	$(CXX) $@ -c $<

clean:
	rm -f *.o
	rm $(BIN)