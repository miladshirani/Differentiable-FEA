# Differentiable-FEA

A hands-on guide to building a finite element method (FEA) solver for large-deformation hyperelastic materials from scratch, using automatic differentiation. This repository walks through setting up FEA for hyperelasticity in parallel using two of the most popular Python autodiff frameworks — **JAX** and **PyTorch** — so you can see both how the method works and how the same ideas are expressed differently across frameworks.

Rather than hand-deriving the stress-strain relation and its tangent (the standard approach, and a common source of bugs in classical FEA codes), every derivative here — shape function gradients, the deformation gradient, the Piola stress, and the consistent tangent stiffness for Newton-Raphson — is computed exactly via autodiff. The goal is to make the mechanics *and* the autodiff mechanics both transparent, one building block at a time.

## Why autodiff for FEA?

In a standard FEA implementation, the constitutive stress-strain relation and its tangent (needed for Newton-Raphson) are derived by hand — a tedious and error-prone step, especially for nonlinear material models. Here, the strain energy density is the only thing defined explicitly; the stress (`∂ψ/∂F`) and tangent stiffness (`∂²ψ/∂F²`) fall out automatically and exactly, with no hand-derived formulas to get wrong.

## Current Features

- **Constitutive model:** Compressible neo-Hookean hyperelasticity, with stress obtained via `jax.grad` of the strain energy
- **Elements:** Tri3, Tri6, Quad4, Quad8, Quad9 (isoparametric shape functions, gradients via `jax.jacfwd`)
- **Integration:** Gauss-Legendre quadrature (1D and 2D, arbitrary order)
- **Kinematics:** Isoparametric interpolation, mesh Jacobian, and both local (reference-element) and global (physical) deformation gradient computed via autodiff through the interpolated displacement field
- **Weak form:** Local and globally-integrated internal virtual work assembly, vectorized over Gauss points with `jax.vmap`

## Roadmap

- [ ] Newton-Raphson solver, with the tangent stiffness matrix obtained via `jax.jacfwd`/`jax.jacrev` on the assembled residual
- [ ] Dynamic Relaxation solver (explicit, matrix-free — robust through buckling/snap-through where Newton-Raphson struggles)
- [ ] Global assembly across a mesh (current code operates at the single-element level)
- [ ] Boundary conditions and load stepping
- [ ] Parallel PyTorch implementation
- [ ] Example problems and convergence/robustness comparison between solvers

## Project Structure

```
neo-Hookean/
└── scr/
    ├── constitutive_relations.py   # Strain energy density and Piola stress (via autodiff)
    ├── shape_functions_2D.py       # Tri3/6, Quad4/8/9 isoparametric shape functions
    ├── Grad_shape_functions_2D.py  # Shape function gradients (via autodiff)
    ├── Gauss_Quadratures.py        # 1D/2D Gauss-Legendre quadrature rules
    ├── Kinematics.py               # Interpolation, mesh Jacobian, deformation gradient
    └── Weak_Form.py                # Local/global weak form assembly
```

## Author

Milad Shirani — Postdoc, Yale University (PhD, UC Berkeley), computational mechanics.
