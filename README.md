# Differentiable-FEA

Differentiable, **matrix-free** finite element analysis for large-strain hyperelasticity, written in
purely functional PyTorch (`torch.func`: `grad`, `jvp`, `vjp`, `vmap`, `jacfwd`). The goal is an open,
reproducible tool for high-performance computing: GPUs first, then distributed memory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miladshirani/Differentiable-FEA/blob/main/notebooks/colab_gpu.ipynb)

> **Status: early development (v0.1.0.dev0).** Today this is a verified **2D, single-process, CPU, float64**
> solver. It is **not yet an HPC code**: the GPU path has not been run on a real GPU, there is no
> distributed-memory support, no scalable preconditioner and no comparison against an established code.
> See [docs/ROADMAP.md](docs/ROADMAP.md) for what is planned and [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)
> for what is known to be broken or unverified. Performance numbers will appear here only when a script in
> `benchmarks/` produced them.

## What works today

| | |
|---|---|
| Elements | Q4, Q8 (serendipity), Q9, Tri3, Tri6 from one element library |
| Geometries | structured rectangle (no mesher needed); rectangle, L-bracket, plate with hole, notched plate via Gmsh |
| Materials | neo-Hookean, Mooney–Rivlin, Saint Venant–Kirchhoff, Demiray, fibre-reinforced HGO; a new model is one energy function |
| Solver | matrix-free inexact Newton–Krylov: PCG on `jvp` products, Eisenstat–Walker forcing, Armijo line search, adaptive load stepping |
| Preconditioners | Jacobi, nodal block-Jacobi (matrix-free); ILU of the assembled tangent (CPU only) |
| Sensitivities | exact adjoint gradients (implicit function theorem), no assembled matrix |
| GUI | Streamlit app (`app/streamlit_app.py`) with live residual monitor |
| Tests | agreement with dolfinx, analytic reference solutions, finite-strain patch test for every element, rigid-body invariance, operator symmetry/SPD, adjoint vs finite differences, material checks; notebook safety scan |

## Validation (measured)

| Check | Result | Reproduce |
|---|---|---|
| **Identical discrete problem vs FEniCSx/dolfinx 0.11.0** (same mesh, neo-Hookean energy, Gauss rule, clamp + dead load, 3 load steps; plate with hole and cantilever; Tri3, Tri6, Q4, Q9) | displacement, strain energy and reaction agree to about 1e-13 relative (worst case 9e-14); a deliberate 0.1 % change of the reference material shows up as 1.7e-4, so the comparison is sensitive | `benchmarks/validate_against_dolfinx.py` -> `benchmarks/results/validate_dolfinx.json` |
| Cantilever vs Timoshenko beam theory (plane strain, small load) | Q8, Q9, Tri6 within 1 % (about 0.4 % below beam theory); Q4 shows the expected shear locking and converges with refinement | `tests/test_analytic.py` |
| Plate with hole vs Heywood's finite-width Kirsch formula | peak stress at the hole within 3 %; Tri6 approaches from below, Q9 from above | `tests/test_analytic.py` |

What this does **not** show: it compares identical discretisations, so it validates the implementation but not
discretisation error or performance, and only one problem family (neo-Hookean, clamp plus traction) was compared.
The tolerances of the analytic tests were set after looking at the measured values, so they document the
current accuracy rather than predict it independently.

## Performance (measured, CPU only so far)

Apple-silicon laptop, 6 threads, float64, Q4 elements. Source: `benchmarks/bench_device.py` and
`benchmarks/profile_matvec.py` (results in `benchmarks/results/`; one run each, repeat timings vary by about 10 %).

| | jvp (reference) | linearize | stored Gauss-point tangent (`qp`, default) | assembled CSR matrix |
|---|---|---|---|---|
| one tangent product, 40k elements | 68 ms | 22 ms | 5.6 ms | 0.61 ms (plus 0.37 s to assemble, 17 MB) |
| complete nonlinear solve, 10k elements (Jacobi-CG, 17 Newton / 2904 CG iterations in all variants) | 72.2 s | - | 11.0 s | - |

* The three matrix-free variants apply the same operator (tests: agreement to 1e-11 for every element and material;
  the two solutions differ by 3e-14). The cost of one product grows linearly with the mesh (2k to 160k elements).
* The original product (`jvp`) reached about 0.3 % of the memory bandwidth; time went into many small unfused
  element-wise operations, not into arithmetic. `qp` reaches about 3 %.
* **On this CPU, in 2D, an assembled sparse matrix is still about 10x faster per product than the best matrix-free
  variant.** Matrix-free is expected to pay off for high-order elements, 3D and on GPUs (memory), not here;
  that has to be measured, and has only been started (see the GPU section).
### First GPU measurement (free Colab T4, float64, two runs)

Source: `benchmarks/results/bench_cuda_t4_colab.json`, `bench_cpu_colab.json`, `profile_cuda_quad4.json` (the Colab CPU
has **one** thread, so GPU-vs-CPU ratios against it say little about a modern multi-core CPU).

| Q4, uniaxial tension | Colab CPU (1 thread) | T4 GPU | laptop CPU (6 threads, from above) |
|---|---|---|---|
| tangent product `qp`, 40k elements | 46-73 ms | 14.8 ms (both runs) | 5.6 ms |
| tangent product `qp`, 160k elements | 184-285 ms | 59 ms (both runs) | 23 ms |
| complete solve, 10k elements, `qp` | 45-49 s | 12.4-12.7 s | 11.0 s |
| complete solve, 10k elements, `jvp` | 201-211 s | 43-45 s | 72.2 s |

* **The CUDA path works**: the same solve on GPU and CPU agrees to 1e-12 (relative, max norm).
* **The T4 is not faster than the laptop CPU here.** The tangent product reaches 0.68 % of the T4's measured memory
  bandwidth (233 GB/s) and its time grows linearly with the mesh, so it is throughput-limited, not launch-limited.
* The profile of the reference `jvp` operator on the T4 puts 74 % of the time into cuBLAS batched double-precision GEMM
  (`volta_dgemm_64x64`) applied to 2x2 / 4x4 matrices: a bad fit. The `qp` operator uses the same kind of batched calls;
  `qp_ew` (same arithmetic with element-wise multiply-and-sum) was added to test that explanation and has **not yet been
  measured on a GPU**. On the CPU it is slower than `qp` (12 vs 5 ms at 40k elements).
* GPU memory: peak 3.9 GB at 160k elements over all operators together (not attributed per operator).

## Design rules

* **Functional, no classes** in the numerical code: state lives in plain dictionaries (`spec` → `params` → `fields`),
  linear operators are closures `v -> A(v)`.
* Per-element work is written for one element and mapped with `vmap`; local–global coupling is one gather and
  one `index_add_`; Dirichlet conditions use masks and `torch.where`.
* Derivatives (stress, tangent products, adjoints) come from automatic differentiation, never from hand-derived formulas.

## Install and quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: pip install -r requirements-lock.txt && pip install -e . --no-deps
python examples/quickstart.py
python -m pytest -q
streamlit run app/streamlit_app.py
```

```python
import torch
from difffea.problem import default_spec, build_problem
from difffea.newton_krylov import newton_krylov_solve
from difffea.postprocess import compute_fields

params = build_problem(default_spec("plate_hole", element_type="tri6", h=0.2))
theta = torch.ones(params["conn"].shape[0], dtype=torch.float64)
u, info = newton_krylov_solve(params, theta, n_load_steps=3, tol=1e-5)
fields = compute_fields(u, theta, params)       # stresses, J, energy, ...
```

## Google Colab

`notebooks/colab_gpu.ipynb` runs the code on Colab's free GPU. It asks for nothing beyond running the
code: no Google Drive, no authentication, no Colab Secrets, no public tunnel. It fetches this repository at a
**pinned commit** and installs only version-pinned packages; `tests/test_notebook_safety.py` (run by CI) scans
every notebook for forbidden patterns. The repository can promise that about the notebook only — it cannot
promise anything about Colab itself. The GPU path is still **unverified** on real hardware.

## Layout

```
src/difffea/   the library (elements, kinematics, materials, operators, newton_krylov, ...)
tests/         pytest suite (+ notebook safety scan)
examples/      quickstart and benchmarks of the predecessor project
app/           Streamlit GUI
notebooks/     Colab notebook
docs/          roadmap, known issues, figures
```

## License

Apache-2.0, see [LICENSE](LICENSE). This project builds on an earlier, private learning project by the same author.
