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
| Tests | finite-strain patch test for every element, rigid-body invariance, operator symmetry/SPD, adjoint vs finite differences, material checks; notebook safety scan |

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
