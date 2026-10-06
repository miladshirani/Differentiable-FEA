"""
make_figures.py
===============

Draws the README figures from the saved result files in benchmarks/results/ (no number is typed in by hand):

    docs/figures/perf_tangent_product.png   time of one tangent product vs mesh size, laptop CPU and T4 GPU
    docs/figures/perf_solve_time.png        time of a complete nonlinear solve, per device and tangent variant
    docs/figures/perf_bandwidth.png         share of the T4's memory bandwidth reached by one tangent product
    docs/figures/validation_dolfinx.png     agreement with FEniCSx/dolfinx, with the sensitivity control

Run (click Run in VS Code, or `python benchmarks/make_figures.py`).  Needs only matplotlib.

Style: one categorical colour per tangent variant, always the same (the first four slots of a documented
colour-blind-checked palette, in its documented order); thin 2 px lines with different marker shapes as a second
encoding; direct value labels only on the points the text talks about; text in neutral ink, never in a data colour.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "docs", "figures")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# one colour + marker per tangent-product variant (same in every figure)
VARIANTS = [("jvp", "jvp (reference)", "#2a78d6", "o"),
            ("linearize", "linearize", "#eb6834", "s"),
            ("qp", "qp (stored tangent, batched matmul)", "#1baf7a", "^"),
            ("qp_ew", "qp_ew (stored tangent, element-wise)", "#eda100", "D")]
KEY = {k: (label, colour, marker) for k, label, colour, marker in VARIANTS}


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def style(ax):
    """Recessive hairline grid, no top/right spine, neutral tick text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=200, facecolor=SURFACE)
    return fig


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


# ------------------------------------------------------------------------------------------ figure 1
def fig_tangent_product():
    """Median time of one A(v) vs number of elements; two panels with a shared time axis."""
    panels = [("Laptop CPU (6 threads)", load("bench_cpu.json")),
              ("NVIDIA T4 GPU (Google Colab)", load("bench_cuda_t4_colab_run2.json"))]
    fig = new_fig(10.2, 4.6)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (title, data) in zip(axes, panels):
        style(ax)
        rows = data["matvec_sweep"]
        x = [r["n_elements"] for r in rows]
        for key, _, colour, marker in VARIANTS:
            y = [r[f"A_{key}_s"] * 1e3 for r in rows if f"A_{key}_s" in r]
            ax.plot(x[:len(y)], y, color=colour, linewidth=2, marker=marker, markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("elements (Q4)", fontsize=9.5, color=INK2)
        ax.set_xticks([2e3, 1e4, 4e4, 1.6e5])
        ax.set_xticklabels(["2k", "10k", "40k", "160k"])
        # direct labels only for the fastest variant of each panel (the one the text talks about)
        fastest = min(KEY, key=lambda k: rows[-1][f"A_{k}_s"] if f"A_{k}_s" in rows[-1] else 9e9)
        ax.annotate(f"{rows[-1][f'A_{fastest}_s'] * 1e3:.1f} ms", (x[-1], rows[-1][f"A_{fastest}_s"] * 1e3),
                    textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color=INK)
    axes[0].set_ylabel("time of one A(v) [ms]", fontsize=9.5, color=INK2)
    handles = [Line2D([0], [0], color=c, linewidth=2, marker=m, markersize=6.5, label=lab)
               for _, lab, c, m in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("One matrix-free tangent product A(v): cost grows linearly with the mesh", x=0.065, ha="left",
                 fontsize=12, color=INK, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.27, wspace=0.08)
    save(fig, "perf_tangent_product.png")


# ------------------------------------------------------------------------------------------ figure 2
def fig_solve_time():
    """Wall time of a complete Newton-Krylov solve (10k Q4 elements, uniaxial tension, Jacobi-CG)."""
    def times(solves):
        return {s["tangent"]: s["seconds"] for s in solves if "note" not in s}
    lap = times(load("bench_cpu.json")["solves"])
    t4 = times(load("bench_cuda_t4_colab_run2.json")["solves"])
    colab = times(load("bench_cpu_colab_run2.json")["solves"])                  # qp, qp_ew
    colab["jvp"] = times(load("bench_cpu_colab.json")["solves"])["jvp"]         # reference solve from run 1
    groups = [("NVIDIA T4 GPU (Colab)", t4), ("Laptop CPU (6 threads)", lap), ("Colab CPU (1 thread)", colab)]

    fig = new_fig(9.0, 4.6)
    ax = fig.subplots()
    style(ax)
    ax.grid(axis="y", visible=False)
    bar_h, gap = 0.20, 0.02
    ticks, labels = [], []
    for g, (name, data) in enumerate(groups):
        keys = [k for k, *_ in VARIANTS if k in data]
        y0 = g * 1.0
        for i, k in enumerate(keys):
            y = y0 + (i - (len(keys) - 1) / 2) * (bar_h + gap)
            ax.barh(y, data[k], height=bar_h, color=KEY[k][1])
            ax.text(data[k] + 3, y, f"{data[k]:.1f} s", va="center", fontsize=9, color=INK)
        ticks.append(y0)
        labels.append(name)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 255)
    ax.set_xlabel("wall time of the complete solve [s]  (lower is better)", fontsize=9.5, color=INK2)
    handles = [Line2D([0], [0], color=c, linewidth=8, label=lab) for k, lab, c, _ in VARIANTS if k != "linearize"]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Complete nonlinear solve, 10,000 Q4 elements (same iteration counts, same solution)",
                 x=0.03, ha="left", fontsize=12, color=INK, y=0.99)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.9, bottom=0.2)
    save(fig, "perf_solve_time.png")


# ------------------------------------------------------------------------------------------ figure 3
def fig_bandwidth():
    """Fraction of the T4's measured memory bandwidth reached by one A(v) (40k elements)."""
    d = load("profile_cuda_t4_colab.json")
    vals = [(k, 100 * d[f"fraction_of_bandwidth_{ {'jvp': 'current'}.get(k, k)}"]) for k, *_ in VARIANTS]
    fig = new_fig(8.2, 3.3)
    ax = fig.subplots()
    style(ax)
    ax.grid(axis="y", visible=False)
    for i, (k, v) in enumerate(vals):
        ax.barh(i, v, height=0.5, color=KEY[k][1])
        ax.text(v + 0.15, i, f"{v:.2f} %", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([KEY[k][0].split(" (")[0] for k, _ in vals], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_xlabel(f"% of the measured {d['bandwidth_GBps_triad']:.0f} GB/s (minimum memory traffic / time)",
                  fontsize=9.5, color=INK2)
    fig.suptitle("T4: how close one tangent product gets to the memory-bandwidth limit", x=0.03, ha="left",
                 fontsize=12, color=INK, y=0.99)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.2)
    save(fig, "perf_bandwidth.png")


# ------------------------------------------------------------------------------------------ figure 4
def fig_validation():
    """Relative difference to dolfinx per case (log axis) and the sensitivity control."""
    d = load("validate_dolfinx.json")
    cases = d["cases"]
    fig = new_fig(9.0, 4.2)
    ax = fig.subplots()
    style(ax)
    ax.grid(axis="y", visible=False)
    for i, c in enumerate(cases):
        ax.plot(max(c["rel_err_u"], 1e-17), i, "o", color="#2a78d6", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
    n = len(cases)
    ctl = d["negative_control"]
    ax.plot(ctl["rel_err_u"], n, "o", color="#eb6834", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(ctl["rel_err_u"] * 0.5, n, "sensitivity control: reference material changed by 0.1 %",
            ha="right", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 1e-2)
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels([c["case"] for c in cases] + ["negative control"], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("max |u_difffea - u_dolfinx| / max |u|   (same mesh, energy, quadrature, loads)", fontsize=9.5,
                  color=INK2)
    fig.suptitle(f"Agreement with FEniCSx/dolfinx {d['dolfinx']}: rounding level, while a 0.1 % change is visible",
                 x=0.03, ha="left", fontsize=12, color=INK, y=0.99)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.88, bottom=0.17)
    save(fig, "validation_dolfinx.png")


if __name__ == "__main__":
    fig_tangent_product()
    fig_solve_time()
    fig_bandwidth()
    fig_validation()
