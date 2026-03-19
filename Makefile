# Makefile
CXX = mpicxx
CXXFLAGS = -O3 -march=native -std=c++14 -fno-math-errno

# 请根据实际安装路径修改以下路径 
INCLUDES = -I$(CONDA_PREFIX)/include/eigen3 -I$(HOME)/software/nlopt/include
LIBS = -L$(HOME)/software/nlopt/lib -lnlopt -lm

TARGET = zibra_mpi_engine
SRC = src/zibra_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS)

clean:
	rm -f $(TARGET)