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

# (clean rule moved below)

# Improved baseline (Phase 1 corrected)
improved:
	$(CXX) -std=c++17 -O0 -I/opt/homebrew/include -I./include $(BENCHMARK_DIR)/main_improved.cpp -o sph_improved $(LDFLAGS)

# 3D simulation with rotatable cube
3d:
	$(CXX) -std=c++17 -O2 -I/opt/homebrew/include -I./include $(BENCHMARK_DIR)/main_3d.cpp -o sph_3d $(LDFLAGS)

# Phase 3: NEON SIMD (-O2 to allow NEON intrinsic optimization)
simd:
	$(CXX) -std=c++17 -O2 -mcpu=apple-m1 -I/opt/homebrew/include -I./include $(BENCHMARK_DIR)/main_simd.cpp -o sph_simd $(LDFLAGS)

# Research Tip: Rule to build with optimizations for comparison later
opt:
	$(CXX) -std=c++17 -O3 $(BENCHMARK_DIR)/main.cpp -o sph_baseline_opt $(LDFLAGS)

# Clean all build artifacts
clean:
	rm -f sph_baseline sph_baseline_opt sph_improved sph_simd sph_3d