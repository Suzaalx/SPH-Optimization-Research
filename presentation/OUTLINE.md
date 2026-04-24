# Presentation Outline — Introduction Talk (5–10 min)

**Title:** Multi-threading vs. SIMD: Optimizing SPH Fluid Simulation on Apple Silicon
**Author:** Sujal Khatri — Fisk University, Department of Computer Science
**Duration target:** 7 minutes (comfortable inside a 5–10 min slot)
**Total slides:** 11

---

## Design guidelines

- Use a clean, minimal template (white background, 1–2 accent colors). The figures in `figures/` are already colored to be consistent.
- Keep each slide to **one big idea + one visual**. Do not read bullets aloud.
- Every data chart is already saved as a PNG in `./figures/`; drop them in directly.
- Suggested title font: Inter / Helvetica Neue, 36 pt. Body: 22–24 pt.
- Footer (optional): `Sujal Khatri · SPH on Apple M1 · 2026`.

---

## Timing map

| Section | Slides | Time |
|---|---|---|
| Hook & context | 1–2 | 1:00 |
| Problem & questions | 3 | 0:45 |
| System architecture | 4–5 | 1:30 |
| Key optimizations | 6–7 | 1:30 |
| Results | 8–9 | 1:30 |
| Takeaways & next steps | 10–11 | 0:45 |
| **Total** | **11** | **≈ 7:00** |

---

## Slide 1 — Title  *(0:20)*

**Visual:** Plain title slide. Optionally include a single still from your 2D dam-break simulation as a subtle background (dim to ~30% opacity).

**What to say:**
> "Hi, I'm Sujal. This project asks a simple question — if you have an Apple M1 chip and you need to simulate fluid in real time, what actually buys you performance: using more cores, using wider vector instructions, or something else entirely? I'll walk you through what I tried and what I learned."

---

## Slide 2 — Why SPH, why performance  *(0:40)*

**Headline:** *Fluids as particles — but at a cost*

**Visual:** A screenshot of your running simulation (dam break, 2D or 3D). On the right, a short bullet block:
- SPH = fluid simulated as thousands of interacting particles
- Each particle needs every other particle for density & forces
- → **O(n²)** per frame. 2,660 particles = ~7M pair evaluations per frame
- To hit 60 FPS, the physics engine has **exactly 16.6 ms** to do all 7 million

**Say:**
> "SPH is an elegant way to simulate fluids — no grid, just particles that push on each other. The catch is that every particle checks every other particle, so the cost grows quadratically. At our test size, that's seven million particle-pair evaluations every single frame. And to hit a smooth 60 FPS, the entire physics engine — density, forces, integration — gets exactly 16.6 milliseconds to finish all of that. That's the budget we're working with, and the baseline blows past it by a factor of 25."

---

## Slide 3 — Research questions  *(0:45)*

**Headline:** *The question I actually wanted to answer*

**Visual:** Plain slide, four questions arranged in two groups with a thin divider between them. Group labels "COMPUTE" and "MEMORY" on the left margin. Q4 stands alone at the bottom as "SYNTHESIS".

**Compute:**
1. How much does **OpenMP multi-threading** speed up SPH on ARM64?
2. What do we gain from **NEON SIMD vectorization**?

**Memory:**
3. How does **AoS vs. SoA** memory layout change the picture?

**Synthesis:**
4. What are the real **architectural bottlenecks** on Apple Silicon?

**Say:**
> "I framed four concrete questions, and they split into two camps. Questions 1 and 2 are about *compute* — can I go faster by using more cores, or by making each core do more per clock cycle? Question 3 is about *memory* — does how I lay data out in RAM change everything? And question 4 is the synthesis: once I've tried all of these, what's actually the wall I'm hitting? That compute-versus-memory divide is going to be the central theme of this talk."

---

## Slide 4 — System architecture  *(0:50)*

**Figure:** `figures/06_system_architecture.png`

**Headline:** *Four versions of the same simulation*

**Say (tracing the diagram with your pointer):**
> "I built four versions of the same simulator so I could compare them fairly. They share the same physics, the same M1 target, and the same Raylib renderer. Phase 1 is the unoptimized baseline. Phase 2 adds OpenMP threading. Phase 3 replaces that with NEON SIMD and a new memory layout. Phase 4 combines both. The bottom row shows where each one landed."

*Tip:* Click to reveal phases one at a time if your slide app supports staged animations.

---

## Slide 5 — Per-frame pipeline  *(0:40)*

**Figure:** `figures/07_frame_pipeline.png`

**Headline:** *Where the time actually goes*

**Say:**
> "Within each frame, there are five steps. Two of them — density and forces — are the O(n²) hot paths that eat almost all the runtime. Those are the targets. Pressure, integration, and rendering are cheap. So everything I did in this project is basically about making steps 1 and 3 as fast as possible."

---

## Slide 6 — Optimization #1: threading  *(0:45)*

**Headline:** *OpenMP: split the outer loop across 8 cores*

**Visual:** A minimal code snippet on the left, a quick architecture note on the right. (No new figure needed, but you can reuse the Phase 2 box of the architecture figure if easier.)

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; ++i) {
    // density / force calc for particle i
}
```

- Each iteration writes only to `particles[i]` → **zero thread contention** — no locks, no atomics, no shared writes
- Apple M1: unified memory means 4 Firestorm + 4 Icestorm cores all read the same physical memory without NUMA penalties
- Compiled with `-O2` and `libomp`

**Say:**
> "The easy win first: threading. The outer loop over particles is embarrassingly parallel — thread 0 computes particle 0's density, thread 1 does particle 1, and so on. The beautiful thing is each thread writes *only* to its own particle, so there is zero contention. No locks. No atomics. No reduction. Just clean, independent work.
>
> And on the M1 specifically, all 8 cores share the same physical memory through Apple's unified memory architecture. That means we avoid the NUMA penalty you'd normally pay on a multi-socket desktop where cores fight over different memory banks. The tradeoff is that the 4 Icestorm efficiency cores run at 2.0 GHz versus 3.2 GHz for Firestorm, so 8 threads gives sub-linear scaling — but still a huge win."

---

## Slide 7 — Optimization #2: SIMD + memory layout  *(0:50)*

**Figure:** `figures/05_aos_vs_soa.png` (top half of slide)
**Optional second figure:** `figures/08_neon_concept.png` (appendix — only if time permits)

**Headline:** *Data layout is half of SIMD*

**Say:**
> "This is the change that mattered most, and it happens *before* I write a single SIMD instruction. With Array of Structures, each particle struct is 32 bytes — position, velocity, force, density, pressure all packed together. A 64-byte cache line fits exactly two of those structs. But the density kernel only needs position — 8 bytes out of 32. So we're paying to ship a full 64-byte box of data across the memory bus and throwing away 48 bytes of it. That's 75% waste on every single cache line load.
>
> Structure of Arrays fixes this completely. Now all the x-positions sit in one contiguous array. One 64-byte cache line gives me 16 consecutive x-values. When NEON's `vld1q_f32` loads 4 floats, they're right there, packed tight, no wasted bytes. *This* is what unlocked the SIMD gains. Without SoA, vectorization is basically pointless because the data isn't there when the vector unit asks for it."

*(If asked later: the SIMD loop uses `vld1q_f32`, `vmlaq_f32`, branchless masks via `vcltq_f32`, and `vaddvq_f32` for horizontal sum. Branching is eliminated entirely using vector compare + bitwise AND.)*

---

## Slide 8 — Results: throughput  *(0:45)*

**Figure:** `figures/01_fps_comparison.png`
**Optional complement:** `figures/02_speedup.png` (if you have room for two side-by-side)

**Headline:** *From 2.56 FPS → 58.89 FPS*

**Say:**
> "Here are the top-line numbers. The baseline drags at two and a half frames per second. OpenMP jumps to 58.89 — that's a 23× speedup. Full transparency: the baseline is deliberately compiled with `-O0`, no compiler optimizations at all, to show the raw algorithmic cost. The optimized versions use `-O2`. So that 23× multiplier includes both OpenMP threading *and* the compiler's own optimizations — loop unrolling, register allocation, instruction scheduling. I want to be upfront about that.
>
> Now, the really interesting thing: look at the SIMD bar. That's a *single thread* on one core at 49.69 FPS — almost as tall as the 8-thread OpenMP bar. That's 85% of the multi-core result on just one core. We'll dig into why on the next slide."

---

## Slide 9 — Results: where the time really goes  *(0:45)*

**Figure:** `figures/03_timing_breakdown.png` on the left, `figures/04_density_win.png` on the right
*(If your slide is too crowded, pick `04_density_win.png` — it's the stronger single image.)*

**Headline:** *The density loop shrinks 5.4× — the force loop barely moves*

**Say:**
> "Looking at raw milliseconds instead of FPS tells the real story. Total physics time drops from 6.2 ms to 3.2 ms per frame. But look at *where* the gains come from. The density loop — 3.87 ms down to 0.72 ms, a 5.4× improvement. The force loop? Basically flat — 2.30 to 2.46 ms.
>
> This isn't random. The density kernel reads *only* positions, so it's the perfect candidate for SoA and vector loads — pure, regular, sequential memory access. The force kernel is a different beast. It reads density, pressure, *and* velocity for every neighbor, it needs square roots and reciprocals, and it has heavy data dependencies between operations. The instruction pipeline can't stay full. That's why SIMD barely helps it — we're hitting the memory wall and the dependency wall at the same time. Understanding *why* one loop moved and the other didn't is what makes this more than just a benchmark chart."

---

## Slide 10 — The takeaway  *(0:30)*

**Figure:** `figures/10_summary.png`

**Headline:** *Memory layout matters as much as parallelism*

**Delivery note:** Slow your cadence way down here. Pause after the 85% number. Let it land. This is the thesis of the entire talk.

**Say:**
 

---

## Slide 11 — What's next & Q&A  *(0:20)*

**Headline:** *Limits & next steps*

Short bulleted list:
- The algorithm is still O(n²) → **spatial hash grid** is the next big win
- Isolate the `-O0 vs -O2` confound with a single-threaded `-O2` baseline *(already disclosed on Slide 8)*
- Hardware counters (Instruments / `perf`) to get hard cache-miss numbers
- GPU acceleration via **Apple Metal**
- Scaling to 10,000+ particles to find where each optimization breaks

**Closing line:**
> "The obvious next step is the one optimization I didn't do: a spatial hash grid, which would turn O(n²) into O(n·k) and dwarf everything I measured. Happy to take questions."

---

# Speaker notes & delivery tips

- **Pace:** Plan to finish around 6:30 so you have buffer. Don't rush Slide 7 (SoA/AoS) or Slide 10 (takeaway) — those are the intellectual climax of the talk.
- **Pointer discipline:** On the architecture slide (Slide 4), physically trace the flow from Phase 1 → 4, especially the transition arrow from the "AoS" block to the "SoA" block. On the pipeline slide (Slide 5), point directly at the O(n²) density and force boxes — make the audience see where 6.2 ms is being spent.
- **One number per slide:** When you mention a number, pause for a beat and let it land. Key numbers: `16.6 ms` (budget), `23×`, `19.4×`, `5.4×`, `85%`, `3.185 ms`.
- **The -O0 disclosure (Slide 8):** You address this *proactively* during results, not in Q&A. Owning the limitation mid-presentation builds credibility. Say it matter-of-factly, not apologetically.
- **The force loop question (Slide 9):** You now explain this in-slide. If pressed further: "The force kernel has a longer dependency chain — it needs the result of a square root before it can compute the pressure gradient, which limits instruction-level parallelism regardless of SIMD width."
- **If asked "why not GPU?":** That's the honest next step. M1 has unified memory, so Metal is low-overhead — but it would change the question from "how do we optimize on CPU" to "when should we leave the CPU entirely."
- **Slide 10 cadence:** This is the most important 30 seconds. Slow down. Let the 85% number breathe. The audience should feel the weight of "one core vs. eight."

---

# Figure index

All figures live in `figures/`. Resolution 200 DPI, white background, ready to drop into slides.

| File | Purpose | Use on slide |
|---|---|---|
| `01_fps_comparison.png`   | Top-line FPS bar chart            | 8 |
| `02_speedup.png`          | Speedup multipliers               | 8 (optional) |
| `03_timing_breakdown.png` | Density + force stacked ms        | 9 |
| `04_density_win.png`      | 5.4× density drop highlight       | 9 |
| `05_aos_vs_soa.png`       | AoS vs SoA cache-line diagram     | 7 |
| `06_system_architecture.png` | Four-phase pipeline overview   | 4 — **Visual note:** ensure the transition arrows guide the eye from the AoS block (P1/P2) to the SoA block (P3/P4). This is the structural turning point. |
| `07_frame_pipeline.png`   | Per-frame physics stages          | 5 — **Visual note:** the O(n²) density and force boxes should be visually heavier (darker fill or thicker border) than the O(n) steps. Point at these during delivery. |
| `08_neon_concept.png`     | Scalar vs 4-wide NEON             | 7 (backup / appendix) |
| `09_scaling.png`          | Baseline O(n²) confirmation       | Appendix / Q&A |
| `10_summary.png`          | Key takeaways infographic         | 10 |

Regenerate any of them by editing `generate_figures.py` and running:

```bash
cd presentation
python3 generate_figures.py
```
