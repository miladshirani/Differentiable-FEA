# Known issues and pitfalls

1. **Gmsh and PyTorch must never share a process.** Both ship an OpenMP runtime; loaded together they
   segfault on larger meshes. Gmsh therefore runs in a child process (`src/difffea/gmsh_worker.py`). Never
   `import gmsh` in the main process. With MPI/distributed runs: mesh on one rank (or offline), then distribute.
2. **float32 does not work with PyTorch 2.14.** `jacfwd`/`jvp` on 0-dim float32 inputs produce float64 internally
   (`grad_shape_functions_2d`, `kinematics`), so float32 runs fail. Apple MPS has no float64, so it is unusable.
   float64 works (also on CUDA). An earlier `jvp`-based workaround did not fix it.
3. **CUDA is unverified.** `spec["device"] = "cuda"` and `to_device` are written and CPU-tested only.
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
