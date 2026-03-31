CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall -fopenmp

TARGET = correlate
SOURCES = main.cpp correlate.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp correlate.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 500 500 4

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean run
