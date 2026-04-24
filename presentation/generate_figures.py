"""
Generate all presentation figures for the SPH optimization research project.
Outputs PNGs into ./figures suitable for dropping into Keynote / PowerPoint / Google Slides.

Run:  python3 generate_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Global style: clean, presentation-friendly
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Consistent color palette across all figures (colorblind-friendly)
C_BASELINE = "#9aa0a6"   # grey
C_OMP      = "#1f77b4"   # blue
C_SIMD     = "#ff7f0e"   # orange
C_COMB     = "#2ca02c"   # green
C_ACCENT   = "#d62728"   # red accent

def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 1: FPS comparison bar chart
# ---------------------------------------------------------------------------
def fig_fps_comparison():
    phases = ["Baseline\n(-O0, AoS)",
              "OpenMP\n(-O2, AoS, 8T)",
              "NEON SIMD\n(-O2, SoA, 1T)",
              "Combined\n(SoA+NEON+OMP)"]
    fps = [2.56, 58.89, 49.69, 57.83]
    colors = [C_BASELINE, C_OMP, C_SIMD, C_COMB]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(phases, fps, color=colors, edgecolor="black", linewidth=0.6)

    ax.axhline(60, color=C_ACCENT, linestyle=":", linewidth=1.6)
    ax.text(0.0, 62.5, "60 FPS display cap", color=C_ACCENT,
            fontsize=11, ha="left")

    for b, v in zip(bars, fps):
        ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Average FPS (higher is better)")
    ax.set_title("SPH Simulation Throughput — 2,660 particles on Apple M1")
    ax.set_ylim(0, 70)
    save(fig, "01_fps_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: Speedup over baseline
# ---------------------------------------------------------------------------
def fig_speedup():
    phases = ["Baseline", "OpenMP", "NEON SIMD", "Combined"]
    speedup = [1.0, 23.0, 19.4, 22.6]
    colors = [C_BASELINE, C_OMP, C_SIMD, C_COMB]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(phases, speedup, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, speedup):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}×",
                ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylabel("Speedup over baseline")
    ax.set_title("Speedup vs. Unoptimized Baseline")
    ax.set_ylim(0, max(speedup) * 1.2)
    save(fig, "02_speedup.png")


# ---------------------------------------------------------------------------
# Figure 3: Per-loop timing breakdown (stacked bars)
# ---------------------------------------------------------------------------
def fig_timing_breakdown():
    phases = ["OpenMP (AoS)", "Combined (SoA + NEON)"]
    density = np.array([3.867, 0.722])
    force   = np.array([2.302, 2.464])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(phases))
    width = 0.55

    b1 = ax.bar(x, density, width, label="Density loop", color=C_OMP,
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, force,   width, bottom=density, label="Force loop",
                color=C_SIMD, edgecolor="black", linewidth=0.5)

    for i, (d, f) in enumerate(zip(density, force)):
        ax.text(i, d/2, f"{d:.3f} ms", ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(i, d + f/2, f"{f:.3f} ms", ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(i, d + f + 0.15, f"Total {d+f:.2f} ms",
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_xticks(x, phases)
    ax.set_ylabel("Time per frame (ms)")
    ax.set_title("Per-Loop Physics Timing (lower is better)")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 8)
    save(fig, "03_timing_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 4: Density loop specifically — highlights the 5.4× SoA win
# ---------------------------------------------------------------------------
def fig_density_win():
    phases = ["OpenMP + AoS", "Combined\n(SoA + NEON + OMP)"]
    times = [3.867, 0.722]
    colors = [C_OMP, C_COMB]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(phases, times, color=colors, edgecolor="black",
                  linewidth=0.6, width=0.55)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width()/2, v + 0.08, f"{v:.3f} ms",
                ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.annotate("", xy=(1, 0.9), xytext=(0, 3.6),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2))
    ax.text(0.5, 2.4, "5.4× faster\nfrom SoA + NEON",
            color=C_ACCENT, fontweight="bold", fontsize=14,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", edgecolor=C_ACCENT, lw=1.5))

    ax.set_ylabel("Density loop time per frame (ms)")
    ax.set_title("Impact of Memory Layout + SIMD on the Density Loop")
    ax.set_ylim(0, 4.6)
    save(fig, "04_density_win.png")


# ---------------------------------------------------------------------------
# Figure 5: AoS vs SoA cache-line diagram
# ---------------------------------------------------------------------------
def fig_aos_vs_soa():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.3))

    field_colors = {
        "pos":  "#2ca02c",
        "vel":  "#1f77b4",
        "frc":  "#ff7f0e",
        "rho":  "#9467bd",
        "prs":  "#e377c2",
    }

    # ---- AoS row ---------------------------------------------------------
    ax = axes[0]
    ax.set_title("Array of Structures (AoS) — 1 cache line = 2 particles (position uses only 25%)",
                 loc="left")
    ax.set_xlim(0, 16); ax.set_ylim(0, 2)
    ax.axis("off")

    fields = [("pos", 2), ("vel", 2), ("frc", 2), ("rho", 1), ("prs", 1)]  # 8 fields per particle = 32B
    x = 0
    for p_idx in range(2):  # 2 particles fit in a 64B cache line
        for name, w in fields:
            ax.add_patch(Rectangle((x, 0.6), w, 0.9,
                                   facecolor=field_colors[name],
                                   edgecolor="black", linewidth=0.6))
            ax.text(x + w/2, 1.05, name, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
            x += w
    # cache line bracket
    ax.plot([0, 16], [0.35, 0.35], color="black", lw=1.5)
    ax.plot([0, 0], [0.25, 0.45], color="black", lw=1.5)
    ax.plot([16, 16], [0.25, 0.45], color="black", lw=1.5)
    ax.text(8, 0.1, "64-byte cache line  →  only 2 particles' positions delivered",
            ha="center", fontsize=11, fontweight="bold")

    # ---- SoA row ---------------------------------------------------------
    ax = axes[1]
    ax.set_title("Structure of Arrays (SoA) — 1 cache line = 16 consecutive pos_x values (100% useful)",
                 loc="left")
    ax.set_xlim(0, 16); ax.set_ylim(0, 2)
    ax.axis("off")

    for i in range(16):
        ax.add_patch(Rectangle((i, 0.6), 1, 0.9,
                               facecolor=field_colors["pos"],
                               edgecolor="black", linewidth=0.6))
        ax.text(i + 0.5, 1.05, f"x{i}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.plot([0, 16], [0.35, 0.35], color="black", lw=1.5)
    ax.plot([0, 0], [0.25, 0.45], color="black", lw=1.5)
    ax.plot([16, 16], [0.25, 0.45], color="black", lw=1.5)
    ax.text(8, 0.1, "64-byte cache line  →  16 pos_x values ready for 4× NEON vector loads",
            ha="center", fontsize=11, fontweight="bold")

    # shared legend
    handles = [mpatches.Patch(color=c, label=n) for n, c in field_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Memory Layout Matters: AoS vs. SoA Cache-Line Utilization",
                 fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "05_aos_vs_soa.png")


# ---------------------------------------------------------------------------
# Figure 6: System architecture — 4-phase optimization pipeline
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="black", text_color="black", fontsize=11, bold=True):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color=text_color)

    def arrow(x1, y1, x2, y2, color="black", lw=1.6, style="->"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, color=color, mutation_scale=18, lw=lw))

    # Title
    ax.text(6.5, 7.15, "System Architecture — 4-Phase Optimization Pipeline",
            ha="center", va="center", fontsize=17, fontweight="bold")

    # Common inputs at top
    box(0.3, 5.9, 3.4, 0.9,
        "Shared Core\n• SPH physics (Müller et al. 2003)\n• 2,660-particle dam break",
        "#eef3ff", fontsize=10)
    box(4.4, 5.9, 3.4, 0.9,
        "Apple M1 target\n• 4 Firestorm + 4 Icestorm cores\n• 128-bit NEON, unified memory",
        "#fff4e6", fontsize=10)
    box(8.5, 5.9, 4.2, 0.9,
        "Raylib rendering (2D / 3D)\n• DrawCircle / GL points\n• identical scene across phases",
        "#f0f7e9", fontsize=10)

    # Phase 1: Baseline
    box(0.3, 3.9, 2.8, 1.4,
        "PHASE 1 — Baseline\n\nAoS layout\n1 thread, -O0\n(reference point)",
        "#eeeeee", fontsize=11)

    # Phase 2: OpenMP
    box(3.4, 3.9, 2.8, 1.4,
        "PHASE 2 — OpenMP\n\nAoS + 8 threads\n#pragma omp parallel for\n-O2",
        "#cfe2f3", fontsize=11)

    # Phase 3: NEON SIMD
    box(6.5, 3.9, 2.8, 1.4,
        "PHASE 3 — NEON SIMD\n\nSoA layout, 1 thread\n4-wide vectors, branchless\n-O2 -mcpu=apple-m1",
        "#fce5cd", fontsize=11)

    # Phase 4: Combined
    box(9.6, 3.9, 2.8, 1.4,
        "PHASE 4 — Combined\n\nSoA + NEON + 8 threads\nOuter loop parallel\nInner loop vectorized",
        "#d9ead3", fontsize=11)

    # Arrows between phases
    arrow(3.1, 4.6, 3.4, 4.6)
    arrow(6.2, 4.6, 6.5, 4.6)
    arrow(9.3, 4.6, 9.6, 4.6)

    # Down arrows to results
    for cx in [1.7, 4.8, 7.9, 11.0]:
        arrow(cx, 3.85, cx, 3.0, color="#555", lw=1.2)

    # Results row
    def result(x, top_text, bottom_text, color):
        box(x, 1.9, 2.8, 1.0,
            top_text + "\n" + bottom_text, color, fontsize=10)

    result(0.3,  "2.56 FPS", "1.0×  (reference)",             "#eeeeee")
    result(3.4,  "58.89 FPS", "23.0× — thread parallelism",   "#cfe2f3")
    result(6.5,  "49.69 FPS", "19.4× — data parallelism",     "#fce5cd")
    result(9.6,  "57.83 FPS", "22.6× — density −5.4×",        "#d9ead3")

    # Bottom banner
    box(0.3, 0.3, 12.4, 1.1,
        "Key measurement: total physics time drops from 6.169 ms (OpenMP) → 3.185 ms (Combined). "
        "Density kernel improves 5.4× from SoA + NEON alone.",
        "#fff8dc", edge="#b8860b", fontsize=12)

    save(fig, "06_system_architecture.png")


# ---------------------------------------------------------------------------
# Figure 7: Per-frame pipeline (what each frame actually does)
# ---------------------------------------------------------------------------
def fig_frame_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.text(6.5, 4.45, "Per-Frame Physics Pipeline",
            ha="center", va="center", fontsize=17, fontweight="bold")

    stages = [
        ("1. Density\nρᵢ = Σ mⱼ W(rᵢⱼ, h)\nPoly6 kernel\nO(n²)", "#cfe2f3"),
        ("2. Pressure\nPᵢ = k(ρᵢ − ρ₀)\nper-particle\nO(n)",    "#d9e2ef"),
        ("3. Forces\npressure + viscosity\n+ gravity\nO(n²)",   "#fce5cd"),
        ("4. Integrate\nsemi-implicit\nEuler + clamp\nO(n)",    "#d9ead3"),
        ("5. Render\nRaylib draw call\nper particle\nO(n)",     "#ead1dc"),
    ]
    n = len(stages)
    gap = 0.15
    total_gap = gap * (n - 1)
    avail = 12.4 - total_gap
    w = avail / n
    x = 0.3
    box_centers = []
    for txt, col in stages:
        ax.add_patch(FancyBboxPatch(
            (x, 1.6), w, 1.9,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=col, edgecolor="black", linewidth=1.2))
        ax.text(x + w/2, 2.55, txt, ha="center", va="center", fontsize=10.5,
                fontweight="bold")
        box_centers.append(x + w/2)
        x += w + gap

    # Under-bracket: steps 1 and 3 are the O(n²) hot paths we optimized
    hot1_lo, hot1_hi = box_centers[0] - w/2, box_centers[0] + w/2
    hot2_lo, hot2_hi = box_centers[2] - w/2, box_centers[2] + w/2
    ax.plot([hot1_lo, hot1_hi], [1.35, 1.35], color=C_ACCENT, lw=2.2)
    ax.plot([hot2_lo, hot2_hi], [1.35, 1.35], color=C_ACCENT, lw=2.2)
    ax.text(box_centers[0], 0.95, "O(n²) hot path\n→ SoA + NEON + OpenMP",
            color=C_ACCENT, fontsize=10.5, ha="center", fontweight="bold")
    ax.text(box_centers[2], 0.95, "O(n²) hot path\n→ OpenMP + partial SIMD",
            color=C_ACCENT, fontsize=10.5, ha="center", fontweight="bold")

    ax.plot([0.3, 12.7], [0.3, 0.3], color="#444", lw=1.2, linestyle="--")
    ax.text(6.5, 0.05, "Outer loop over i — parallel across threads   |   Inner loop over j — 4-wide NEON vectors",
            color="#444", fontsize=10.5, ha="center", style="italic")

    save(fig, "07_frame_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 8: NEON inner-loop concept (scalar vs. 4-wide)
# ---------------------------------------------------------------------------
def fig_neon_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Scalar
    ax = axes[0]
    ax.set_title("Scalar loop\n(1 particle / iteration)", loc="center", fontsize=13)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")
    for i, label in enumerate(["j", "j+1", "j+2", "j+3"]):
        y = 4.5 - i*1.1
        ax.add_patch(Rectangle((1, y), 1.2, 0.8,
                               facecolor="#cfe2f3", edgecolor="black"))
        ax.text(1.6, y + 0.4, label, ha="center", va="center",
                fontweight="bold")
        ax.annotate("", xy=(5, y + 0.4), xytext=(2.3, y + 0.4),
                    arrowprops=dict(arrowstyle="->", color="#555"))
        ax.add_patch(Rectangle((5, y), 3.5, 0.8,
                               facecolor="#eeeeee", edgecolor="black"))
        ax.text(6.75, y + 0.4, "sub, mul, add (1 lane)",
                ha="center", va="center", fontsize=10)

    # NEON
    ax = axes[1]
    ax.set_title("NEON vector loop\n(4 particles / iteration)", loc="center", fontsize=13)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")
    colors = ["#cfe2f3", "#d9e2ef", "#fce5cd", "#d9ead3"]
    for i, (label, col) in enumerate(zip(["j", "j+1", "j+2", "j+3"], colors)):
        ax.add_patch(Rectangle((1, 4.5 - i*0.9), 1.2, 0.75,
                               facecolor=col, edgecolor="black"))
        ax.text(1.6, 4.5 - i*0.9 + 0.375, label, ha="center", va="center",
                fontweight="bold")
    ax.annotate("", xy=(5, 4.0), xytext=(2.3, 4.0),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2))
    ax.add_patch(Rectangle((5, 2.6), 4.5, 2.8,
                           facecolor="#fff4e6",
                           edgecolor=C_ACCENT, linewidth=2))
    ax.text(7.25, 4.7, "vld1q_f32  — load 4 lanes", ha="center", fontsize=11)
    ax.text(7.25, 4.1, "vsubq / vmulq / vmlaq_f32", ha="center", fontsize=11)
    ax.text(7.25, 3.5, "vcltq_f32 + vandq_u32  (branchless)",
            ha="center", fontsize=11)
    ax.text(7.25, 2.9, "vaddvq_f32  — horizontal sum",
            ha="center", fontsize=11)
    ax.text(7.25, 1.8, "1 instruction ≈ 4 scalar operations",
            ha="center", fontsize=12, fontweight="bold",
            color=C_ACCENT)

    fig.suptitle("Inner-Loop Vectorization Concept",
                 fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "08_neon_concept.png")


# ---------------------------------------------------------------------------
# Figure 9: Baseline O(n^2) scaling
# ---------------------------------------------------------------------------
def fig_scaling():
    n = np.array([1500, 2660])
    fps_measured = np.array([6.6, 2.4])            # midpoints of reported ranges
    # Theoretical: fps ∝ 1/n^2
    fps_theoretical = fps_measured[0] * (n[0] / n) ** 2

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(n, fps_measured, "o-", color=C_OMP, lw=2,
            markersize=10, label="Measured (baseline)")
    ax.plot(n, fps_theoretical, "s--", color=C_ACCENT, lw=2,
            markersize=9, label="Theoretical O(n²)")
    for xi, y in zip(n, fps_measured):
        ax.annotate(f"{y:.1f} FPS", (xi, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold")

    ax.set_xlabel("Particle count (n)")
    ax.set_ylabel("Average FPS")
    ax.set_title("Baseline Scaling — O(n²) Confirmed")
    ax.legend(frameon=True)
    ax.set_xlim(1200, 3000)
    ax.set_ylim(0, 8)
    save(fig, "09_scaling.png")


# ---------------------------------------------------------------------------
# Figure 10: Summary "one-slide takeaway" infographic
# ---------------------------------------------------------------------------
def fig_summary():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(6, 6.1, "Key Takeaways", ha="center", va="center",
            fontsize=20, fontweight="bold")

    items = [
        ("23.0×",  "OpenMP 8T speedup\n(threading + -O2)",           C_OMP),
        ("19.4×",  "Single-core NEON SIMD\n≈ 85% of 8-thread OMP",   C_SIMD),
        ("5.4×",   "Density loop drop\nfrom AoS → SoA + NEON",       C_COMB),
        ("3.185 ms", "Combined per-frame\nphysics time",             C_ACCENT),
    ]
    w = 2.7
    for i, (big, small, color) in enumerate(items):
        x = 0.3 + i * (w + 0.2)
        ax.add_patch(FancyBboxPatch(
            (x, 2.0), w, 3.3,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="white", edgecolor=color, linewidth=2.2))
        ax.text(x + w/2, 4.3, big, ha="center", va="center",
                fontsize=28, fontweight="bold", color=color)
        ax.text(x + w/2, 2.9, small, ha="center", va="center",
                fontsize=12)

    ax.add_patch(FancyBboxPatch(
        (0.3, 0.3), 11.4, 1.4,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#fff8dc", edgecolor="#b8860b", linewidth=1.4))
    ax.text(6, 1.0,
            "Lesson: memory layout matters as much as parallelization —\n"
            "one well-optimized core on SoA nearly matches 8 cores on AoS.",
            ha="center", va="center", fontsize=14, fontweight="bold")

    save(fig, "10_summary.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_fps_comparison()
    fig_speedup()
    fig_timing_breakdown()
    fig_density_win()
    fig_aos_vs_soa()
    fig_architecture()
    fig_frame_pipeline()
    fig_neon_concept()
    fig_scaling()
    fig_summary()
    print("\nAll figures written to", OUT)
