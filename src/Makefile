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

# 3D simulation with rotatable cube (NEON SIMD optimized)
3d:
	$(CXX) -std=c++17 -O2 -mcpu=apple-m1 -I/opt/homebrew/include -I./include $(BENCHMARK_DIR)/main_3d.cpp -o sph_3d $(LDFLAGS)

# Phase 2: OpenMP multi-threaded (-O2, links Homebrew libomp)
omp:
	$(CXX) -std=c++17 -O2 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/include -I./include $(BENCHMARK_DIR)/main_omp.cpp -o sph_omp $(LDFLAGS) -L/opt/homebrew/opt/libomp/lib -lomp

# Phase 3: NEON SIMD (-O2 to allow NEON intrinsic optimization)
simd:
	$(CXX) -std=c++17 -O2 -mcpu=apple-m1 -I/opt/homebrew/include -I./include $(BENCHMARK_DIR)/main_simd.cpp -o sph_simd $(LDFLAGS)

# Research Tip: Rule to build with optimizations for comparison later
opt:
	$(CXX) -std=c++17 -O3 $(BENCHMARK_DIR)/main.cpp -o sph_baseline_opt $(LDFLAGS)

# Clean all build artifacts
clean:
	rm -f sph_baseline sph_baseline_opt sph_improved sph_omp sph_simd sph_3d