# Compiler and Flags
CXX = clang++
# -O0 disables optimizations to ensure a "True Baseline" for research
CXXFLAGS = -std=c++17 -O0 -I/opt/homebrew/include -I./include
LDFLAGS = -L/opt/homebrew/lib -lraylib -framework OpenGL -framework Cocoa -framework IOKit

# Target Folders
BENCHMARK_DIR = benchmark
COMMON_DIR = common

# Output Binary Name
TARGET = sph_baseline

# Default build rule
all: $(TARGET)

$(TARGET): $(BENCHMARK_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(BENCHMARK_DIR)/main.cpp -o $(TARGET) $(LDFLAGS)

# Rule to clean build files
clean:
	rm -f $(TARGET)

# Research Tip: Rule to build with optimizations for comparison later
opt:
	$(CXX) -std=c++17 -O3 $(BENCHMARK_DIR)/main.cpp -o sph_baseline_opt $(LDFLAGS)