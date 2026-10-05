# Roadmap

Order chosen so that every claim rests on something measured. "Done" means verified by tests or a script in `benchmarks/`.

| # | Step | Status |
|---|------|--------|
| 0 | Package layout (`src/difffea`), license, CI, notebook safety scan, secret scan | in progress |
| 1 | Validation: analytic solutions (Kirsch, patch tests) and comparison with FEniCSx/dolfinx | planned |
| 2 | Profile on CPU, then verify the GPU path on a Colab T4; honest CPU-vs-GPU timings | planned |
| 3 | 3D hexahedra (Hex8/Hex20), where matrix-free and GPUs pay off | planned |
| 4 | Scalable preconditioner: Chebyshev smoother, p-multigrid, geometric/algebraic multigrid | planned |
| 5 | Distributed memory: mesh partitioning, halo exchange, distributed CG (`torch.distributed`, gloo on a laptop first) | planned |
| 6 | Multi-GPU (NCCL) once hardware is available | planned, hardware needed |
| 7 | Physics: mixed/F-bar, plasticity (non-symmetric tangent → GMRES), contact | planned |
| 8 | Release quality: reproducible benchmarks, docs, pinned releases | continuous |

Design constraint for step 5: the communicator is a dictionary of closures (`all_reduce`, `halo_exchange`),
not a class, so the numerical code stays functional; `torch.distributed` keeps the process group.
