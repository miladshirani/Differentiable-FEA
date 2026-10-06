# Roadmap

Order chosen so that every claim rests on something measured. "Done" means verified by tests or a script in `benchmarks/`.

| # | Step | Status |
|---|------|--------|
| 0 | Package layout (`src/difffea`), license, CI, notebook safety scan, secret scan | done locally; CI not yet run on GitHub |
| 1 | Validation: analytic solutions (beam, Kirsch/Heywood) and comparison with FEniCSx/dolfinx | done for the cases listed in the README; other geometries/materials/load types not yet compared |
| 2 | Profile on CPU, then verify the GPU path on a Colab T4; honest CPU-vs-GPU timings | CPU profile, a 6.5x faster tangent product and the first T4 measurements done; GPU product is far from bandwidth-bound, element-wise variant `qp_ew` awaiting its GPU measurement |
| 3 | 3D hexahedra (Hex8/Hex20), where matrix-free and GPUs pay off | planned |
| 4 | Scalable preconditioner: Chebyshev smoother, p-multigrid, geometric/algebraic multigrid | planned |
| 5 | Distributed memory: mesh partitioning, halo exchange, distributed CG (`torch.distributed`, gloo on a laptop first) | planned |
| 6 | Multi-GPU (NCCL) once hardware is available | planned, hardware needed |
| 7 | Physics: mixed/F-bar, plasticity (non-symmetric tangent → GMRES), contact | planned |
| 8 | Release quality: reproducible benchmarks, docs, pinned releases | continuous |

Design constraint for step 5: the communicator is a dictionary of closures (`all_reduce`, `halo_exchange`),
not a class, so the numerical code stays functional; `torch.distributed` keeps the process group.
