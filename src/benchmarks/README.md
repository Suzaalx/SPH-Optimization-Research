# SPH benchmark logs

Each run writes a log when you close the window: `sph_run_YYYYMMDD_HHMMSS.log`.

## How to read the log

- **Config**: Simulation parameters (particle count, h, density, viscosity, etc.).
- **Run**: Total frames and average FPS (lower FPS = heavier run, e.g. more particles).
- **Final particle bounds**: `x_min`, `x_max`, `y_min`, `y_max` — where the fluid ended up.

## Log review (satisfactory or not)

**Example:** `sph_run_20260305_195516.log`

- **Particles**: 1500  
- **Final bounds**: x ∈ [10, 790], y ∈ [586, 590]  
- **Interpretation**: Particles stayed within the walls (x 10–790, y ≤ 590). They settled at the bottom (y ≈ 586–590), which is expected after a dam break.  
- **Verdict**: Satisfactory — fluid settled correctly, no escape, flat bottom surface.

**What to check for**

- **Satisfactory**: `x_min` ≥ wall margin (e.g. 10), `x_max` ≤ width − margin (e.g. 790), `y_max` ≤ height − margin (e.g. 590). Particles form a puddle at the bottom.
- **Problem**: Particles outside these bounds, or huge bounds, or very low FPS with no visible motion.
