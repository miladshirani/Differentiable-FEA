# Known issues and pitfalls

1. **Gmsh and PyTorch must never share a process.** Both ship an OpenMP runtime; loaded together they
   segfault on larger meshes. Gmsh therefore runs in a child process (`src/difffea/gmsh_worker.py`). Never
   `import gmsh` in the main process. With MPI/distributed runs: mesh on one rank (or offline), then distribute.
2. **float32 does not work with PyTorch 2.14.** `jacfwd`/`jvp` on 0-dim float32 inputs produce float64 internally
   (`grad_shape_functions_2d`, `kinematics`), so float32 runs fail. Apple MPS has no float64, so it is unusable.
   float64 works (also on CUDA). An earlier `jvp`-based workaround did not fix it.
3. **CUDA is verified only on one GPU** (a Colab Tesla T4, float64, Q4/tension problems; solution equals the CPU one to 2.4e-12). Other GPUs, multi-GPU, the ILU/Gmsh paths on GPU and large 3D problems are untested.
4. **Preconditioning is the main algorithmic gap.** Point-Jacobi needs thousands of CG iterations on quadratic or
   bending-dominated meshes; nodal block-Jacobi barely helps; ILU is excellent but CPU-only and its memory
   grows faster than linear. A scalable (multigrid-type) preconditioner is missing.
5. Slender structures need adaptive load stepping; fixed load schedules stall.
6. Streamlit caches imported modules: restart the server after editing the library.
7. Plane strain only, displacement-only formulation (volumetric locking as ν → 0.5), dead loads, no plasticity.
8. CI workflow (`.github/workflows/ci.yml`) has not run yet; gmsh on the Linux runner is untested.
9. Q4 (full 2x2 integration) shows shear locking in bending (measured: tip deflection 4 % too small at 4 elements
   per beam height). Use quadratic elements, or a future B-bar/F-bar formulation.
10. The dolfinx comparison covers Tri3, Tri6, Q4 and Q9 only (dolfinx has no 8-node serendipity geometry), a
    clamp-plus-traction load and the neo-Hookean material. Other materials and boundary conditions are untested against it.
11. macOS can mark files inside `.venv` as hidden, and Python ignores hidden `.pth` files, so an editable install
    (`pip install -e .`) may suddenly stop importing `difffea` (`ModuleNotFoundError`). Fix: `chflags -R nohidden .venv`
    (or set `PYTHONPATH=src`). The test-suite is immune (`pythonpath = ["src", "tests"]` in `pyproject.toml`).
12. `torch.func.linearize` prints a harmless `get_attr Node` UserWarning (inside PyTorch).
13. The tangent moduli stored by the default operators (`qp`, `qp_ew`) take about 128 bytes per Gauss point (512 B per Q4 element);
    use `tangent="linearize"` or `"jvp"` if memory is tight. GPU memory per variant has not been measured separately.
14. The Gmsh wheel needs the system library `libGLU.so.1`. Colab does not have it (Gmsh meshing fails there with
    `OSError: libGLU.so.1`), so the Colab notebook uses the structured mesher. On Ubuntu: `sudo apt-get install libglu1-mesa`.
    The CI workflow installs it, but that workflow has not run yet.
