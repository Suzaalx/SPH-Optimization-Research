# Presentation Assets

Materials for the 5–10 minute introduction talk on the SPH optimization project.

## Contents

- `OUTLINE.md` — slide-by-slide script, timing map, and speaker notes.
- `generate_figures.py` — one script that produces every figure used in the talk.
- `figures/` — rendered PNGs (200 DPI, white background) ready to drop into Keynote / PowerPoint / Google Slides.

## Regenerate figures

```bash
cd presentation
python3 generate_figures.py
```

Requires only `matplotlib` and `numpy`. Edit colors, titles, or values directly in `generate_figures.py`.

## Quick figure index

| File                              | What it shows                                              |
|-----------------------------------|------------------------------------------------------------|
| `01_fps_comparison.png`           | Baseline → OpenMP → SIMD → Combined FPS bars               |
| `02_speedup.png`                  | Same data as speedup multipliers                           |
| `03_timing_breakdown.png`         | Density vs. force time (ms) for OMP vs. Combined           |
| `04_density_win.png`              | Highlights the 5.4× density-loop improvement               |
| `05_aos_vs_soa.png`               | Cache-line diagram explaining why SoA matters              |
| `06_system_architecture.png`      | Four-phase optimization pipeline                           |
| `07_frame_pipeline.png`           | Per-frame physics stages and which are parallelized        |
| `08_neon_concept.png`             | Scalar vs. 4-wide NEON inner loop                          |
| `09_scaling.png`                  | Baseline O(n²) scaling sanity check                        |
| `10_summary.png`                  | One-slide takeaway infographic                             |

## Suggested slide order

See `OUTLINE.md` for the full script. Abridged map:

1. Title
2. Why SPH, why performance
3. Research questions
4. System architecture — **fig 06**
5. Per-frame pipeline — **fig 07**
6. Optimization: threading (code snippet)
7. Optimization: SIMD + SoA — **fig 05** (+ optional **fig 08**)
8. Results: throughput — **fig 01** (+ optional **fig 02**)
9. Results: per-loop timing — **fig 03** + **fig 04**
10. Takeaways — **fig 10**
11. Limitations & next steps
