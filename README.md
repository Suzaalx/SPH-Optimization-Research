# SPH-ARM64: Multi-threading vs. SIMD Performance Analysis

> **Comparative Research Project** on Instruction-Level and Thread-Level Parallelism in Fluid Simulations using Apple Silicon (M1).

## 📋 Project Overview

This repository contains a **research-focused implementation of Smoothed Particle Hydrodynamics (SPH)** optimized for modern ARM64 architectures. The primary objective is to scientifically benchmark and analyze how different optimization strategies—ranging from scalar baseline code to vectorized SIMD and multi-threaded implementations—impact performance in particle-based fluid simulations.

### Research Questions
- How much speedup can we achieve with **multi-threading (OpenMP)** on ARM64?
- What are the performance gains from **SIMD vectorization (ARM NEON)**?
- How do **memory access patterns** (AoS vs. SoA) affect cache efficiency?
- What are the architectural bottlenecks in particle-based simulations?

By systematically comparing these implementation strategies, this study quantifies speedup factors, identifies performance bottlenecks (cache misses, branch mispredictions, memory bandwidth), and provides insights into efficient scientific computing on Apple Silicon.

---

## 🔬 Theoretical Foundation

The physics implementation follows the **2014 Eurographics State-of-the-Art Report (STAR)** on SPH Fluids and **Müller et al.'s foundational work** on particle-based fluid simulation. The simulation approximates the **Navier-Stokes equations** by interpolating fluid properties (density, pressure, velocity) across discrete particles using kernel functions.

### Core Physics Algorithm

**Step 1: Density Calculation** (O(n²) bottleneck)
$$\rho_i = \sum_{j=1}^{N} m_j W(|\mathbf{r}_i - \mathbf{r}_j|, h)$$

Each particle samples neighboring particles within smoothing radius $h$ to compute local density.

**Step 2: Pressure Computation**
$$P_i = k(\rho_i - \rho_0)$$

Using the ideal gas law to convert density to pressure, where $k$ is the gas stiffness constant and $\rho_0$ is rest density.

### Step 3: Force Accumulation
The total force acting on particle $i$ combines the pressure gradient, viscosity, and external gravity forces:

$$\mathbf{F}_i = -\sum_{j} m_j \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} \right) \nabla W(r_{ij}, h) + \mu \sum_{j} m_j \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W(r_{ij}, h) + m_i \mathbf{g}$$

### Step 4: Time Integration
We use a semi-implicit Euler integration to update the velocity and position of each particle:

$$\mathbf{v}_{t+\Delta t} = \mathbf{v}_t + \frac{\mathbf{F}_t}{m} \Delta t$$

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{v}_{t+\Delta t} \Delta t$$

Semi-implicit Euler integration for numerical stability.

### Key Physics Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Smoothing Radius (h) | 16.0 | Kernel support radius |
| Rest Density (ρ₀) | 1.0 | Reference fluid density |
| Gas Stiffness (k) | 2000.0 | Pressure response to density changes |
| Viscosity (μ) | 250.0 | Fluid friction coefficient |
| Particle Mass (m) | 1.0 | Individual particle mass |

---

## 🎯 Research Methodology & Development Phases

The project is structured into **four distinct phases**, each isolating specific optimization techniques to measure their individual impact:

### **Phase 1: Baseline (Control Group)**
- **Architecture:** Single-threaded, scalar C++ code
- **Memory Layout:** Array of Structures (AoS) – intuitive but cache-unfriendly
- **Particle Count:** ~1,500 particles
- **Target Metric:** Establish O(n²) complexity overhead and baseline frame rate
- **Expected Performance:** ~15-25 FPS (baseline)

**Why AoS is slow:** Loading density calculations only needs position data, but AoS forces the CPU to also load velocity and force into cache—wasting memory bandwidth.

### **Phase 2: Task Parallelism (OpenMP)**
- **Architecture:** Multi-core parallelization using OpenMP pragmas
- **Execution Model:** Distribute density/force loops across M1 Performance cores
- **Memory Layout:** Still AoS (to isolate parallelism benefits)
- **Target Metric:** Measure thread-scaling efficiency and "memory wall" effects
- **Expected Speedup:** 2-3x (limited by memory bandwidth contention)

**Challenge:** Multiple cores requesting the same data simultaneously causes memory bus saturation.

### **Phase 3: Data Parallelism (ARM NEON SIMD)**
- **Architecture:** Single-core, 128-bit vectorization using ARM NEON intrinsics
- **Memory Layout:** Refactored to Structure of Arrays (SoA)
  - Instead of `[Particle1, Particle2, Particle3...]`
  - Use `[pos_x[], pos_y[], vel_x[], vel_y[]...]`
- **Instruction Set:** `vld1q_f32`, `vmulq_f32`, `vaddq_f32` (NEON 4-wide float operations)
- **Target Metric:** Instruction-level parallelism and throughput improvements
- **Expected Speedup:** 3-4x per core

**Benefit:** Load 4 floats in a single instruction; process 4 particles' positions simultaneously.

### **Phase 4: Combined Optimization**
- **Architecture:** SoA-based SIMD code running across all M1 cores
- **Execution:** Vectorized loops + OpenMP multi-threading
- **Target Metric:** Measure scaling of vectorized code under multi-core load
- **Expected Speedup:** 6-10x (combined effect)

---

## 🖥️ Hardware Environment

| Component | Details |
|-----------|---------|
| **CPU** | Apple M1 (ARMv8-A ISA) |
| **Performance Cores** | 4 × Firestorm (3.2 GHz) |
| **Efficiency Cores** | 4 × Icestorm (2.0 GHz) |
| **L1 Instruction Cache** | 192 KB |
| **L1 Data Cache** | 128 KB |
| **L2 Cache** | 4 MB (per core) |
| **Memory Architecture** | Unified Memory (CPU/GPU shared) |
| **Compiler** | clang++ (C++17) |

**Why ARM64 matters for this research:**
- ARM NEON provides 128-bit vector operations (4 floats per instruction)
- Unified memory simplifies data movement between cores
- Apple Silicon's in-order execution model makes optimization patterns more predictable

---

## 📁 Directory Structure

```
SPH-Optimization-Research/
├── src/
│   ├── benchmark/
│   │   └── main.cpp              # Current baseline implementation
│   ├── Makefile                  # Build system for all phases
│   └── (future phases)
├── docs/
│   └── baseline documentation    # Physics explanations and algorithm details
├── README.md                     # This file
└── .gitignore
```

### Building & Running

```bash
# Navigate to src directory
cd src

# Build baseline
make

# Run the simulation
./sph_baseline

# Build optimized version (for comparison)
make opt
./sph_baseline_opt

# Clean build artifacts
make clean
```

---

## 📚 Research References

This project is grounded in peer-reviewed computational physics literature:

1. **SPH Fluids in Computer Graphics: State of the Art Report (2014)**
   - Koschier et al., Eurographics
   - [PDF](https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf)
   - Provides comprehensive SPH formulation and modern optimizations

2. **Particle-Based Fluid Simulation for Interactive Applications (2003)**
   - Müller, Charypar, Gross
   - [PDF](https://matthias-research.github.io/pages/publications/sca03.pdf)
   - Foundational work on real-time SPH with practical implementation details

---

## 🔍 What's Happening in the Code

### Baseline Algorithm Flow

1. **Initialization:** 1,500 particles randomly placed in upper region (dam break scenario)
2. **Density Loop:** For every particle pair (i, j), compute kernel contribution—this is the O(n²) bottleneck
3. **Pressure Calculation:** Convert density to pressure using stiff equation of state
4. **Force Calculation:** Accumulate pressure, viscosity, and gravity forces
5. **Integration:** Update velocities and positions using semi-implicit Euler
6. **Boundary Handling:** Bounce particles off floor (y > 580)
7. **Rendering:** Display particles as blue circles using Raylib

### Performance Metrics to Track

- **Frame Rate (FPS):** Inversely proportional to per-frame computation
- **Density Loop Time:** Most expensive phase (O(n²))—primary optimization target
- **Cache Miss Rate:** Measure with Instruments or `perf` on Linux
- **Memory Bandwidth:** Analyze with ARM Streamline Profiler

---

## 🎓 Learning Outcomes

By completing this research project, you'll understand:

✅ How **particle-based simulations** approximate fluid dynamics  
✅ The difference between **task parallelism** (threads) and **data parallelism** (SIMD)  
✅ Why **memory layout matters** more than raw algorithm complexity  
✅ How to profile and optimize **compute-bound workloads**  
✅ ARM NEON **intrinsic programming** and compiler optimizations  
✅ Real-world constraints: **memory bandwidth**, **cache hierarchy**, **thermal limits**  

---

## 📊 Expected Results

| Implementation | Est. Frame Rate | Speedup vs. Baseline | Primary Bottleneck |
|---|---|---|---|
| Phase 1: Baseline | 20 FPS | 1.0x | CPU computation |
| Phase 2: OpenMP | 50 FPS | 2.5x | Memory bandwidth |
| Phase 3: NEON SIMD | 70 FPS | 3.5x | Single-core limits |
| Phase 4: Combined | 150+ FPS | 7.5x+ | Thermal/frequency |

(Actual results will vary based on particle count and hardware thermal state)

---

## 🚀 Next Steps

- [ ] Complete Phase 2 (OpenMP parallelization)
- [ ] Implement Phase 3 (NEON vectorization with SoA refactoring)
- [ ] Add performance profiling infrastructure (timing loops, event counters)
- [ ] Generate benchmark comparison plots
- [ ] Document compiler optimization flags and their impact
- [ ] Explore GPU acceleration as Phase 5
