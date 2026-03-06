# How to Implement SIMD (ARM NEON) in the SPH Code

This guide explains how to add ARM NEON vectorization to the baseline SPH implementation. The main steps are: **refactor to SoA**, then **vectorize the inner (j) loop** in the density and force passes.

---

## 1. High-level strategy

| Step | What to do |
|------|------------|
| **SoA layout** | Replace `std::vector<Particle>` with separate arrays so we can load 4 floats at a time (e.g. 4 x-positions). |
| **Vectorize inner loop** | For each particle `i`, process neighbors `j` in chunks of 4 using NEON (4-wide float). |
| **Keep physics identical** | Same kernels and formulas; only the data layout and loop structure change. |

NEON is 128-bit: **4 × float** per instruction. We use it in the **j-loop** (over neighbors) because that’s where we do many similar operations (load positions, compute r², accumulate density).

---

## 2. Refactor to Structure of Arrays (SoA)

**Current (AoS):**
```cpp
struct Particle {
    glm::vec2 position;   // x, y
    glm::vec2 velocity;   // vx, vy
    glm::vec2 force;      // fx, fy
    float density;
    float pressure;
};
std::vector<Particle> particles(n);
```

**SoA for SIMD:**
```cpp
// SoA: one array per component, length n
std::vector<float> pos_x(n), pos_y(n);
std::vector<float> vel_x(n), vel_y(n);
std::vector<float> force_x(n), force_y(n);
std::vector<float> density(n), pressure(n);
```

- **Why:** For the inner loop we need `pos_x[j]`, `pos_x[j+1]`, … for four consecutive `j`. With AoS, those x values are not adjacent in memory. With SoA, `pos_x[j..j+3]` is one contiguous vector load.
- **Rendering:** Each frame, either iterate SoA and call `DrawCircle(pos_x[i], pos_y[i], ...)` or temporarily pack into a small AoS buffer for the draw loop—your choice.

---

## 3. Include NEON and align data (optional but helpful)

```cpp
#include <arm_neon.h>

// Optional: align SoA buffers for better load/store
alignas(16) std::vector<float> pos_x(n), pos_y(n);
// ... same for other arrays
```

Use `alignas(16)` if your standard library supports it for `std::vector` element buffer; otherwise ensure you don’t cross cache lines badly when loading 4 floats.

---

## 4. Vectorize the density inner loop (j)

**Scalar idea (unchanged physics):**
```text
for i in 0..n:
  density[i] = 0
  for j in 0..n:
    r_x = pos_x[j] - pos_x[i],  r_y = pos_y[j] - pos_y[i]
    r_sq = r_x*r_x + r_y*r_y
    if r_sq < h2:  density[i] += mass * poly6(r_sq)
  pressure[i] = gasConst * (density[i] - restDensity)
```

**Vectorized j-loop (process 4 j’s at a time):**

- Load 4 `pos_x[j]` and 4 `pos_y[j]` with `vld1q_f32(&pos_x[j])` and `vld1q_f32(&pos_y[j])`.
- Broadcast particle `i`’s position: `pos_x[i]` and `pos_y[i]` into two `float32x4_t` (all 4 lanes same).
- Compute 4 r_vectors and 4 r_squared:
  - `r_x = pos_x_j - pos_x_i`, `r_y = pos_y_j - pos_y_i` (vector subtract).
  - `r_sq = r_x*r_x + r_y*r_y` (vector mul/add).
- **Mask:** compare `r_sq < h2` (vector compare), then use the result to blend kernel contribution with zero (so only neighbors inside radius count).
- **Kernel:** For each of the 4 r_squared values, compute `diff = h2 - r_sq`, then `contrib = mass * poly6 * diff^3`. Do this with vector ops (vector mul, then horizontal sum or keep as 4 contributions).
- **Accumulate:** Add the 4 contributions to `density[i]` (horizontal sum of a float32x4_t, then add to scalar).

**Minimal NEON sketch for one chunk of 4 j’s (density):**

```cpp
#include <arm_neon.h>

void density_step_simd(const float* pos_x, const float* pos_y,
                       float* density, float* pressure,
                       int n, const SPHSettings& s) {
    const float32x4_t vh2 = vdupq_n_f32(s.h2);
    const float32x4_t vpoly6 = vdupq_n_f32(s.mass * s.poly6);

    for (int i = 0; i < n; i++) {
        float32x4_t vxi = vdupq_n_f32(pos_x[i]);
        float32x4_t vyi = vdupq_n_f32(pos_y[i]);
        float sum = 0.0f;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t vxj = vld1q_f32(pos_x + j);
            float32x4_t vyj = vld1q_f32(pos_y + j);
            float32x4_t rx = vsubq_f32(vxj, vxi);
            float32x4_t ry = vsubq_f32(vyj, vyi);
            float32x4_t r2 = vaddq_f32(vmulq_f32(rx, rx), vmulq_f32(ry, ry));

            // mask: r2 < h2
            uint32x4_t mask = vcltq_f32(r2, vh2);
            float32x4_t diff = vsubq_f32(vh2, r2);
            float32x4_t d3 = vmulq_f32(vmulq_f32(diff, diff), diff);
            float32x4_t contrib = vmulq_f32(vpoly6, d3);
            contrib = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(contrib), mask));

            sum += vaddvq_f32(contrib);  // horizontal sum (ARMv8); else use vget_lane + vadd
        }
        for (; j < n; j++) { /* scalar remainder */ }

        density[i] = sum;
        pressure[i] = s.gasConstant * (sum - s.restDensity);
    }
}
```

- **Remainder:** After the `j + 4 <= n` loop, add a small scalar loop for `j` from the last multiple of 4 to `n`, reusing your existing scalar `poly6_kernel` and accumulation into `sum`.
- **Horizontal sum:** `vaddvq_f32` is available on ARMv8. If you target ARMv7, replace with two `vget_low_f32`/`vget_high_f32` and add the lanes manually.

This is the pattern: **SoA + inner j-loop in chunks of 4 with NEON, scalar i-loop and remainder**.

---

## 5. Force loop (same idea, more ops)

- Keep SoA: `force_x[i]`, `force_y[i]`, and for each j load `pos_x[j], pos_y[j]`, `vel_x[j], vel_y[j]`, `density[j]`, `pressure[j]`.
- For each chunk of 4 j’s:
  - Compute `r_x`, `r_y`, `r_sq`, and **r_length** = `sqrt(r_sq)` (vector sqrt: `vsqrtq_f32`).
  - Compute 4 pressure terms and 4 viscosity terms (same math as scalar, in vector form).
  - Compute 4 × (fx, fy) contributions, mask by `r_sq < h2 && r_sq > eps`, then **horizontal sum** the 4 fx and the 4 fy into `force_x[i]` and `force_y[i]`.
- Again, do a scalar remainder for the last 1–3 j’s.

The force loop is more involved (sqrt, more temporaries, two accumulators) but the principle is the same: **SoA + vectorized j-loop in chunks of 4**.

---

## 6. Integration and boundaries

- Integration stays scalar or can be vectorized in i (e.g. process 4 particles at a time: load 4 density, 4 force_x, etc., update 4 vel_x, 4 pos_x). For a first SIMD version, scalar is fine.
- Boundaries: keep your current logic; run it per particle (or vectorize by loading 4 positions/velocities, doing 4 comparisons, and storing back).

---

## 7. Build and test

- **Compiler:** clang or gcc for ARM64 (Apple Silicon: default clang is fine).
- **Flags:** `-mcpu=apple-m1` or `-march=armv8-a+simd` so NEON is enabled.
- **Check:** Run the same scenario (same n, h, dt, seed) and compare final particle bounds (and optionally a few density/force values) between scalar and SIMD to ensure they match closely (small numerical differences are OK).

---

## 8. Summary

| Task | Action |
|------|--------|
| **Data layout** | SoA: `pos_x[]`, `pos_y[]`, `vel_x[]`, `vel_y[]`, `force_x[]`, `force_y[]`, `density[]`, `pressure[]`. |
| **Density loop** | Outer loop over `i` (scalar); inner loop over `j` in chunks of 4 with NEON (load 4 positions, compute 4 r², mask, kernel, horizontal sum into `density[i]`). |
| **Force loop** | Same: vectorize inner j-loop in chunks of 4; accumulate `force_x[i]` and `force_y[i]` with horizontal sums. |
| **Remainder** | After vector loop, scalar loop for remaining j (and i if you vectorize integration). |
| **Rendering** | Use `pos_x[i]`, `pos_y[i]` from SoA in your draw loop (or convert to AoS once per frame if you prefer). |

Once this is in place, you can add **OpenMP** on the outer **i-loop** (e.g. `#pragma omp parallel for`) to get both SIMD and multi-threading (Phase 4).
