# How to Implement Multi-threading (OpenMP) in the SPH Code

This guide explains how to add multi-threading using **OpenMP** to the baseline SPH implementation. The idea: parallelize the **outer loop over particles (i)** in the density, force, and integration steps so each thread works on a different range of particles. No change to the physics or data layout is required.

---

## 1. Why the outer loop is safe to parallelize

In our current code:

- **Density step:** For each `i`, you only **write** to `particles[i].density` and `particles[i].pressure`. You **read** all `particles[j]` (positions). So different `i` values touch different memory; no two threads write the same element.
- **Force step:** Same: each `i` only **writes** `particles[i].force`; reads are from all `j`.
- **Integration:** Each iteration only **writes** one particle’s velocity and position.

So parallelizing the **outer loop over `i`** (and the per-particle loops for clamp and integration) is **data-race free**. No locks or atomics are needed.

---

## 2. Which loops to parallelize

| Step | Loop | OpenMP |
|------|------|--------|
| Density | `for (i = 0; i < n; i++)` | Yes – parallelize this outer loop. |
| Density clamp | `for (auto& p : particles)` | Yes – parallel for over index. |
| Force | `for (i = 0; i < n; i++)` | Yes – parallelize outer loop. |
| Integration | `for (auto& p : particles)` | Yes – parallel for over index. |
| Rendering | Draw loop | **No** – keep on main thread (Raylib/OpenGL). |

Only the **computational** loops are parallelized; the **draw** loop stays single-threaded.

---

## 3. Code changes

### 3.1 Include OpenMP header

At the top of the file (e.g. after standard headers):

```cpp
#include <omp.h>
```

### 3.2 Density loop

**Before:**
```cpp
for (size_t i = 0; i < particles.size(); i++) {
    particles[i].density = 0.0f;
    for (size_t j = 0; j < particles.size(); j++) {
        // ...
    }
    particles[i].pressure = settings.gasConstant * (particles[i].density - settings.restDensity);
}
```

**After:**
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < particles.size(); i++) {
    particles[i].density = 0.0f;
    for (size_t j = 0; j < particles.size(); j++) {
        // ... unchanged ...
    }
    particles[i].pressure = settings.gasConstant * (particles[i].density - settings.restDensity);
}
```

- `schedule(static)` splits the `i` range in contiguous chunks between threads (good for cache when each thread works on its own range of particles). You can try `schedule(dynamic)` if load is uneven; often `static` is enough.

### 3.3 Density clamp loop

Use an index loop so OpenMP can parallelize it:

**Before:**
```cpp
for (auto& p : particles) {
    if (p.density < settings.restDensity * 0.1f)
        p.density = settings.restDensity * 0.1f;
}
```

**After:**
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < particles.size(); i++) {
    if (particles[i].density < settings.restDensity * 0.1f)
        particles[i].density = settings.restDensity * 0.1f;
}
```

### 3.4 Force loop

**Before:**
```cpp
for (size_t i = 0; i < particles.size(); i++) {
    particles[i].force = gravity;
    for (size_t j = 0; j < particles.size(); j++) {
        // ...
    }
}
```

**After:**
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < particles.size(); i++) {
    particles[i].force = gravity;
    for (size_t j = 0; j < particles.size(); j++) {
        // ... unchanged ...
    }
}
```

### 3.5 Integration loop

**Before:**
```cpp
for (auto& p : particles) {
    if (p.density > 0.0001f) {
        p.velocity += (p.force / p.density) * dt;
        p.position += p.velocity * dt;
    }
    handle_boundaries(p, WINDOW_WIDTH, WINDOW_HEIGHT);
}
```

**After:**
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < particles.size(); i++) {
    if (particles[i].density > 0.0001f) {
        particles[i].velocity += (particles[i].force / particles[i].density) * dt;
        particles[i].position += particles[i].velocity * dt;
    }
    handle_boundaries(particles[i], WINDOW_WIDTH, WINDOW_HEIGHT);
}
```

Rendering stays as-is (no `#pragma omp` on the draw loop).

---

## 4. Build with OpenMP

**Apple Clang (macOS):**

Apple’s clang does not ship with OpenMP. Use one of:

- **Homebrew LLVM** (recommended): install `llvm` and use its `clang++` with OpenMP:
  ```bash
  brew install libomp llvm
  # Then use the compiler that supports -fopenmp, e.g.:
  /opt/homebrew/opt/llvm/bin/clang++ -fopenmp -std=c++17 -O2 ... benchmark/main_improved.cpp -o sph_omp ...
  ```
- **MacPorts:** install `libomp` and use the compiler that supports `-fopenmp`.

**GCC or Clang on Linux:**

```bash
clang++ -fopenmp -std=c++17 -O2 ...   # or g++ -fopenmp ...
```

**Makefile snippet:**

```makefile
# Optional: build with OpenMP (requires LLVM or GCC with OpenMP)
CXX_OPENMP = /opt/homebrew/opt/llvm/bin/clang++
OPENMP_FLAGS = -fopenmp

sph_omp: benchmark/main_improved.cpp
	$(CXX_OPENMP) $(CXXFLAGS) $(OPENMP_FLAGS) benchmark/main_improved.cpp -o sph_omp $(LDFLAGS)
```

Adjust `CXX_OPENMP` to your OpenMP-capable compiler.

---

## 5. Controlling number of threads

- **Default:** OpenMP usually uses all logical cores. You can set the number of threads **before** the parallel region (e.g. at the start of `main()` or before the first `#pragma omp parallel`):
  ```cpp
  #include <omp.h>
  int main() {
      omp_set_num_threads(4);   // use 4 threads (e.g. M1 P-cores)
      // ...
  }
  ```
- **Environment variable:** `export OMP_NUM_THREADS=4` (no code change).
- **In code:** Call `omp_set_num_threads(N)` once at startup. Useful for experiments (e.g. 1 vs 2 vs 4 threads).

---

## 6. What to expect

- **Speedup:** Typically **~2–3×** on 4 cores for this kind of O(n²) loop, often limited by **memory bandwidth** (all threads reading the same particle array).
- **Scaling:** Speedup may level off as you add threads; that indicates a bandwidth or cache bottleneck rather than a bug.
- **Correctness:** Results (particle positions, final bounds) should match the single-threaded run; only wall-clock time and FPS should change.

---

## 7. Optional: avoid including OpenMP when not available

If you want one codebase that compiles with or without OpenMP:

```cpp
#ifdef _OPENMP
#include <omp.h>
#define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
#define OMP_PARALLEL_FOR
#endif

// Then in the loops:
OMP_PARALLEL_FOR
for (size_t i = 0; i < particles.size(); i++) {
    // ...
}
```

- With `-fopenmp`: `_OPENMP` is defined and the pragma is used.
- Without OpenMP: the macro is empty and the loop runs single-threaded.

---

## 8. Summary

| Step | Action |
|------|--------|
| Add OpenMP | `#include <omp.h>`, add `#pragma omp parallel for schedule(static)` on the **outer** `i` loop in density, clamp, force, and integration. |
| Keep rendering single-threaded | Do **not** put OpenMP on the draw loop. |
| Build | Use a compiler with OpenMP (`-fopenmp`) and link OpenMP if needed (e.g. LLVM’s clang on macOS). |
| Tune | Use `omp_set_num_threads(N)` or `OMP_NUM_THREADS` to measure scaling (1, 2, 4 threads). |

After this, we can combine with the SIMD (NEON) version: same OpenMP parallelization on the outer loop, with the inner j-loop vectorized (Phase 4).
