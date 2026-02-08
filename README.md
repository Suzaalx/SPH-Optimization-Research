# SPH-ARM64: Multi-threading vs. SIMD Performance Analysis
> **Comparative Research Project** on Instruction-Level and Thread-Level Parallelism in Fluid Simulations using Apple Silicon (M1).

## 1. Project Overview
This repository contains a research-focused implementation of **Smoothed Particle Hydrodynamics (SPH)**. The primary goal is to benchmark and analyze how different optimization strategies perform on modern ARM64 architectures. 

By comparing a scalar **Baseline**, an **OpenMP Multi-threaded** version, and an **ARM NEON SIMD** version, this study quantifies the speedup factors and architectural bottlenecks (cache misses, branch mispredictions, and memory bandwidth) inherent in particle-based simulations.

## 2. Theoretical Foundation
The physics implementation follows the **2014 Eurographics State-of-the-Art Report (STAR)** on SPH Fluids. The simulation approximates the Navier-Stokes equations by interpolating properties across a set of discrete particles using a kernel function $W(r, h)$.

### Core Mathematical Steps:
1. **Density Summation:** $\rho_i = \sum_{j} m_j W(r_{ij}, h)$
2. **Pressure Calculation:** Based on the Ideal Gas Law or Tait's Equation.
3. **Internal Forces:** Calculating the Pressure Gradient and Viscosity Laplacian.
4. **Integration:** Semi-implicit Euler or Leapfrog integration for motion.



## 3. Research Methodology & Phases
The project is structured into four distinct development phases to isolate the performance impact of each optimization:

### Phase 1: The Baseline (Control)
* **Execution:** Single-threaded, scalar math.
* **Memory Layout:** Array of Structures (AoS).
* **Objective:** Establish the "slow" reference point and measure initial $O(n^2)$ complexity overhead.

### Phase 2: Task Parallelism (OpenMP)
* **Execution:** Multi-core (utilizing M1 Performance cores).
* **Focus:** Measuring thread-scaling efficiency and the impact of the "memory wall" when multiple cores request data simultaneously.

### Phase 3: Data Parallelism (SIMD/NEON)
* **Execution:** Single-core, 128-bit Vectorization.
* **Refactor:** Transition to **Structure of Arrays (SoA)** to allow `vld1q_f32` (NEON load) instructions to fetch four floats at once.
* **Objective:** Maximize throughput per clock cycle.

### Phase 4: Combined Optimization
* **Execution:** Vectorized code running across all available CPU threads.



## 4. Hardware Environment
* **CPU:** Apple M1 (ARMv8-A Architecture)
* **Cores:** 4 Performance (Firestorm) / 4 Efficiency (Icestorm)
* **L1 Cache:** 192 KB Instruction, 128 KB Data
* **Memory:** Unified Memory Architecture (UMA)

## 5. Directory Structure
```text
.
├── src/
│   ├── baseline/      # Phase 1: Initial C++ implementation
│   ├── parallel/      # Phase 2: OpenMP / std::thread code
│   └── vectorized/    # Phase 3: ARM NEON Intrinsic implementation
├── benchmarks/        # Performance logs (CSV) and Python plotting scripts
├── docs/              # Research references and 2014 EG STAR paper
└── Makefile           # Central build system