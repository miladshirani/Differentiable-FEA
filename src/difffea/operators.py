"""
operators.py
============

From ONE element to the WHOLE mesh, and from a residual to LINEAR OPERATORS.

This file is the heart of the matrix-free method.  It contains

    elem_residual            internal force of one element, straight from virtual work
                             (the REFERENCE implementation: slow, maximally transparent)
    precompute_geometry      hoist the loop-invariant geometry (dN/dX, w*detJ) out of Newton
    elem_residual_precomputed  the same element force, using the cached geometry (FAST)
    global_residual          gather -> vmap(element force) -> scatter-add -> mask
    make_A_operator          v -> J(u) v      (forward-mode AD, jvp)
    make_AT_operator         v -> J(u)^T v    (reverse-mode AD, vjp)
    compute_jacobi_diagonal  diag(J) without ever forming J
    make_jacobi_preconditioner   r -> r / diag(J)

The ``params`` dictionary (a "Pytree")
--------------------------------------
    nodes    (N, 2)        reference node coordinates X
    conn     (n_el, nen)   node ids of every element
    shape_fn callable      shape functions of the element type
    GP, GW   (n_q, 2),(n_q,) Gauss points / weights
    material str, mat_p (n_p,), mat_scalable (n_p,)  material model, nominal parameters, which
                           parameters the design field theta scales (see materials.py)
    [mu0, lmbda0 float     legacy neo-Hookean parameters, used if 'material' is absent]
    f        (2N,)         external force vector for load factor 1
    m        (2N,)         Dirichlet mask: 1.0 free DOF, 0.0 constrained DOF
    fixed    (2N,) bool    True on constrained DOFs
    u_D      (2N,)         prescribed values on constrained DOFs (for load factor 1)
    dN_dX    (n_el, n_q, nen, 2)  [optional cache] shape-function gradients w.r.t. X
    w_detJ   (n_el, n_q)          [optional cache] Gauss weight * det(dX/dxi)
"""

import torch
from torch.func import grad, jvp, vjp, vmap
from difffea.kinematics import Coordinate_Interpolation, Displacement_Interpolation, Mesh_Jacobian
from difffea.grad_shape_functions_2d import Grad_Shape_Functions_2D
from difffea.materials import psi_2d, material_of, element_material_parameters
from difffea.weak_form import Integrated_Global_Virtual_Work

def elem_residual(ue, Xe, mu, lmbda, shape_fn, Gauss_Points, Gauss_Weights):
    """
    Computes the nodal internal force vector R^e for ONE element.

    Parameters
    ----------
    ue : Tensor of shape (nen, 2)
        Physical nodal displacements of this element
    Xe : Tensor of shape (nen, 2)
        Undeformed nodal coordinates of this element
    mu, lmbda : float or Tensor
        Material parameters
    shape_fn : Callable
        Shape function (e.g. shape_fn_quad4)
    Gauss_Points, Gauss_Weights : Tensors
        Gauss points (num_gp, 2) and weights (num_gp,)

    Returns
    -------
    Tensor of shape (nen, 2)
        Internal nodal forces acting on each node of the element.
    """
    # 1. Create spatial interpolation closures from nodal values
    X_fn = Coordinate_Interpolation(shape_fn, Xe)
    u_fn = Displacement_Interpolation(shape_fn, ue)

    # 2. Define virtual work strictly as a function of the virtual nodal vector (dv)
    def virtual_work(dv):
        # Interpolate the test/virtual field from nodal variations dv
        v_fn = Displacement_Interpolation(shape_fn, dv)
        return Integrated_Global_Virtual_Work(u_fn, v_fn, X_fn, mu, lmbda, Gauss_Points, Gauss_Weights)

    # 3. Taking the gradient with respect to dv gives R^e directly!
    # Evaluated at dv = 0 (shape: (nen, 2))
    return grad(virtual_work)(torch.zeros_like(ue))




# =============================================================================
# HOISTING LOOP INVARIANTS: geometry does not change during the solve
# =============================================================================
def precompute_geometry(params):
    """
    Compute, ONCE per mesh, the quantities that depend only on the undeformed
    geometry and therefore never change during Newton iterations:

        dN_dX[e, q, a, :]  = grad_X N_a at Gauss point q of element e
                           = (dX/dxi)^{-T} grad_xi N_a              (chain rule)
        w_detJ[e, q]       = w_q * det(dX/dxi)  (physical quadrature weight)

    With these two arrays the element force needs NO shape-function evaluation,
    NO Jacobian and NO autodiff of geometry any more -- only the stress law.

    Returns a NEW dictionary (functional style: the input is not modified)
    containing everything in ``params`` plus the two cached arrays.
    """
    nodes, conn = params["nodes"], params["conn"]
    shape_fn, GP, GW = params["shape_fn"], params["GP"], params["GW"]

    def geometry_at_point(X_fn, xi, eta):
        # reference gradients dN/d(xi, eta): (nen, 2)
        dN_dxi = Grad_Shape_Functions_2D(shape_fn, xi, eta)
        # mesh Jacobian dX/d(xi, eta), its determinant and inverse
        _, det_J, inv_J = Mesh_Jacobian(X_fn, xi, eta)
        # chain rule  dN_a/dX_i = sum_j dN_a/dxi_j (J^-1)_{ji}      -> (nen, 2)
        return torch.einsum("aj,ji->ai", dN_dxi, inv_J), det_J

    def geometry_of_element(Xe):
        X_fn = Coordinate_Interpolation(shape_fn, Xe)
        # vmap over the Gauss points of the element
        dN_dX, det_J = vmap(lambda xi, eta: geometry_at_point(X_fn, xi, eta))(
            GP[:, 0], GP[:, 1])                       # (n_q, nen, 2), (n_q,)
        return dN_dX, GW * det_J

    # vmap over all elements: gather coordinates (n_el, nen, 2) and map
    dN_dX, w_detJ = vmap(geometry_of_element)(nodes[conn])
    return {**params, "dN_dX": dN_dX, "w_detJ": w_detJ}


def get_geometry(params):
    """Cached geometry if ``precompute_geometry`` was applied, else computed on the fly."""
    if "dN_dX" in params:
        return params["dN_dX"], params["w_detJ"]
    cached = precompute_geometry(params)
    return cached["dN_dX"], cached["w_detJ"]


def elem_residual_precomputed(ue, dN_dX, w_detJ, p, psi_fn):
    """
    Internal force of ONE element from the cached geometry.  This is the
    discrete virtual work written out explicitly:

            R[a, i] = sum_q  w_q detJ_q  P_ij(F_q)  dN_a/dX_j (q)

    Parameters
    ----------
    ue     : (nen, 2)       nodal displacements of the element
    dN_dX  : (n_q, nen, 2)  cached shape gradients
    w_detJ : (n_q,)         cached physical weights
    p      : (n_p,)         material parameters of this element
    psi_fn : callable       plane-strain energy  psi(F, p)  (``Materials.psi_2d``)

    Returns
    -------
    (nen, 2)  internal nodal forces (identical to ``elem_residual`` for neo-Hookean)
    """
    # displacement gradient at every Gauss point: du_i/dX_j = sum_a u_{a,i} dN_a/dX_j
    grad_u = torch.einsum("ai,qaj->qij", ue, dN_dX)                       # (n_q, 2, 2)
    F = torch.eye(2, dtype=ue.dtype, device=ue.device) + grad_u           # (n_q, 2, 2)
    # stress at every Gauss point (autodiff of the energy, vectorised over q)
    P = vmap(grad(psi_fn, argnums=0), in_dims=(0, None))(F, p)            # (n_q, 2, 2)
    # quadrature of  P : grad(N_a e_i)  ->  (nen, 2)
    return torch.einsum("q,qij,qaj->ai", w_detJ, P, dN_dX)


def global_residual(u, theta, params, load_factor=1.0):
    """
    Computes the assembled global out-of-balance residual vector R(u).

    Parameters
    ----------
    u : Tensor of shape (2 * num_nodes,)
        Flat global nodal displacement vector
    theta : Tensor of shape (num_elements,)
        Element stiffness/density scaling factors
    params : dict
        Problem dictionary containing mesh, connectivity, boundary conditions, etc.
    load_factor : float
        Current load continuation scaling factor in [0, 1]

    Returns
    -------
    Tensor of shape (2 * num_nodes,)
        Masked global residual vector R(u)
    """
    element_connectivity = params["conn"]          # Shape: (num_elements, nodes_per_element)
    nodes = params["nodes"]                        # Shape: (num_nodes, 2)
    num_nodes = nodes.shape[0]
    num_elements, nodes_per_element = element_connectivity.shape

    # 1. Gather coordinates and displacements for all elements:
    # Shape of both: (num_elements, nodes_per_element, 2)
    nodal_displacements = u.reshape(num_nodes, 2)
    element_displacements = nodal_displacements[element_connectivity]

    # 2. Material parameters per element
    # (n_el, n_p) table of material parameters (design field theta scales the stiffness ones)
    element_p = element_material_parameters(theta, params)
    psi_fn = psi_2d(material_of(params)[0])

    # 3. Vectorize the element force over ALL elements in parallel via pure vmap.
    #    Per-element arguments (axis 0): displacements, cached geometry, material.
    dN_dX, w_detJ = get_geometry(params)           # (n_el, n_q, nen, 2), (n_el, n_q)
    element_residuals = vmap(
        elem_residual_precomputed,
        in_dims=(0, 0, 0, 0, None)
    )(
        element_displacements,
        dN_dX,
        w_detJ,
        element_p,
        psi_fn
    )
    # element_residuals shape: (num_elements, nodes_per_element, 2)

    # 4. Global Assembly: Scatter-Add into global internal force array
    flat_connectivity = element_connectivity.reshape(-1)
    flat_element_forces = element_residuals.reshape(-1, 2)

    R_internal = torch.zeros(num_nodes, 2, dtype=u.dtype, device=u.device)
    R_internal.index_add_(0, flat_connectivity, flat_element_forces)

    # 5. Out-of-balance residual: R = R_internal - load * F_external
    R_global = R_internal.reshape(-1) - load_factor * params["f"]

    # 6. Apply Dirichlet projection mask (residual = 0 on fixed boundaries)
    return params["m"] * R_global







# =============================================================================
# 1. FORWARD TANGENT OPERATOR: A(v) = J(u) * v
# =============================================================================
def make_A_operator(u, theta, params):
    """
    PURPOSE:
    --------
    Constructs the matrix-free forward linear operator A(v) = J(u) * v.
    
    ARGUMENTS:
    ----------
    u      : (2 * num_nodes,) Current displacement state where physics is linearized.
    theta  : (num_elements,) Element stiffness/density scaling factors (design variables).
    params : Problem dictionary containing 'm' (Dirichlet mask), mesh data, etc.

    HOW IT RELATES TO OTHER FUNCTIONS:
    -----------------------------------
    - Calls `global_residual` (from Step 4) via forward-mode AD (`torch.func.jvp`).
    - Handed directly to the linear solver `pcg` (Step 6) to compute search directions.
    """
    m = params["m"]  # 1.0 for Free DOFs, 0.0 for Dirichlet (clamped) DOFs
    
    def A(v):
        """
        Matrix-free operator evaluation: A(v) = m * J(u)*(m*v) + (1 - m)*v
        v has shape (2 * num_nodes,).
        """
        # 1. Why (m * v,)?
        #    The Krylov solver's search vector 'v' has arbitrary numbers everywhere,
        #    including on clamped boundary nodes! Multiplying (m * v) zeroes out
        #    perturbations on fixed boundaries so we don't calculate fictitious strains.
        # 2. Why does jvp compute the Jacobian-vector product?
        #    jvp evaluates: d/d(eps) [ R(u + eps * (m*v)) ] at eps=0.
        #    It computes J(u) * (m*v) in a single forward pass without forming J!
        _, Jv = jvp(
            lambda w: global_residual(w, theta, params),
            (u,),
            (m * v,)
        )
        
        # 3. Why m * Jv + (1.0 - m) * v?
        #    - Free DOFs (m == 1.0): Returns the true mechanical stiffness Jv.
        #    - Clamped DOFs (m == 0.0): Returns 1.0 * v = v (uncoupled identity pivot).
        #    This prevents zero eigenvalues (which crash Conjugate Gradient) while
        #    maintaining strict symmetry: A = M*J*M + (I - M) = A^T.
        return m * Jv + (1.0 - m) * v

    return A


# =============================================================================
# 1b. FASTER FORWARD TANGENT OPERATORS (same mathematics, different cost structure)
# =============================================================================
# ``make_A_operator`` re-evaluates the whole residual (and the stress derivative inside it) at EVERY
# Krylov iteration, although u does not change during the linear solve.  Profiling (benchmarks/
# profile_matvec.py) shows that this costs 3x more than necessary, and that the time goes into
# many small, unfused element-wise tensor operations -- not into arithmetic or memory bandwidth.
#
#   "jvp"        make_A_operator              recompute everything each product        (reference)
#   "linearize"  make_A_operator_linearize    evaluate the primal ONCE per Newton step, then only
#                                             propagate the tangent                    (same result)
#   "qp"         make_A_operator_qp           store the tangent moduli C = d^2 psi/dF^2 at every
#                                             Gauss point once per Newton step; one product is then
#                                             gather -> small batched matmul -> scatter-add
#                                             ("partial assembly"; no global matrix, but ~128 B of
#                                             storage per Gauss point)
#   "qp_ew"      make_A_operator_qp_ew        same as "qp" with element-wise multiply-and-sum instead of
#                                             batched GEMM (the tiny matrices make GEMM inefficient on GPUs)
#
# All three return a closure v -> A(v) with identical semantics (Dirichlet identity pivot included).
def make_A_operator_linearize(u, theta, params):
    """
    A(v) = m*J(u)*(m*v) + (1-m)*v via ``torch.func.linearize``: the residual and everything it needs at
    the linearisation point is evaluated ONCE here; each call of the returned closure only pushes the
    tangent vector through the stored linearised graph.  Mathematically identical to ``make_A_operator``.
    """
    from torch.func import linearize
    m = params["m"]
    _, jvp_fn = linearize(lambda w: global_residual(w, theta, params), u)

    def A(v):
        """Jv through the stored linearisation; v has shape (2N,)."""
        return m * jvp_fn(m * v) + (1.0 - m) * v

    return A


def _qp_setup(u, theta, params):
    """
    Shared set-up of the stored-tangent operators: the Gauss-point tangent moduli at u.

    Returns (C, dN_dX) with
        C     (n_el, n_q, 4, 4)  C[e, q, (ij), (kl)] = w_q detJ_q * d^2 psi / (dF_ij dF_kl)   (4x4 matrix per point)
        dN_dX (n_el, n_q, nen, 2)  cached shape-function gradients
    The second derivative is computed by autodiff (``hessian`` of the plane-strain energy of the chosen
    material), vmapped over Gauss points (shared material parameters) and elements.
    """
    conn = params["conn"]
    N = params["nodes"].shape[0]
    n_el = conn.shape[0]
    dN_dX, w_detJ = get_geometry(params)                       # (n_el, n_q, nen, 2), (n_el, n_q)
    n_q = dN_dX.shape[1]
    element_p = element_material_parameters(theta, params)     # (n_el, n_p)
    psi_fn = psi_2d(material_of(params)[0])

    # deformation gradient at every Gauss point: F = I + sum_a u_a (x) dN_a/dX      (n_el, n_q, 2, 2)
    ue = u.reshape(N, 2)[conn]                                 # (n_el, nen, 2)
    F = torch.eye(2, dtype=u.dtype, device=u.device) + torch.einsum("eai,eqaj->eqij", ue, dN_dX)

    hess = torch.func.hessian(psi_fn, argnums=0)               # (F (2,2), p) -> (2, 2, 2, 2)
    C = vmap(vmap(hess, in_dims=(0, None)), in_dims=(0, 0))(F, element_p)      # (n_el, n_q, 2, 2, 2, 2)
    # fold the quadrature weight in and flatten (ij) and (kl) so that C acts as a 4x4 matrix
    C = (C * w_detJ[:, :, None, None, None, None]).reshape(n_el, n_q, 4, 4)
    return C, dN_dX


def make_A_operator_qp(u, theta, params):
    """
    Tangent product with the material tangent STORED at the Gauss points (batched-matmul version).

    With C from ``_qp_setup``, one product A(v) is, per element e and Gauss point q:

            grad_v   = sum_a v_a (x) dN_a/dX                         gather + einsum   (n_el, n_q, 2, 2)
            dP       = C : grad_v                                    batched 4x4 matvec
            f[a, i]  = sum_q dP_ij dN_a/dX_j                         einsum
            Jv       = scatter-add of f                              one index_add_

    which is exactly J v for the discrete virtual work of ``elem_residual_precomputed`` (the geometric and
    material parts of the tangent are both inside C because F = I + grad u is the argument of psi).

    Memory: 16 numbers per Gauss point (128 B in float64), e.g. 512 B per Q4 element.  Use
    ``make_A_operator_linearize`` when that is too much.

    NOTE (measured on a Tesla T4): the einsum/matmul calls dispatch to cuBLAS batched GEMM with 64x64 tiles
    for 2x2 / 4x4 matrices, which wastes the GPU; ``make_A_operator_qp_ew`` does the same arithmetic with
    plain element-wise operations.
    """
    conn, m = params["conn"], params["m"]
    N = params["nodes"].shape[0]
    n_el = conn.shape[0]
    C, dN_dX = _qp_setup(u, theta, params)
    n_q = dN_dX.shape[1]
    flat_conn = conn.reshape(-1)

    def A(v):
        """Jv from the stored moduli; v has shape (2N,)."""
        ve = (m * v).reshape(N, 2)[conn]                                       # (n_el, nen, 2), masked
        dgrad = torch.einsum("eai,eqaj->eqij", ve, dN_dX).reshape(n_el, n_q, 4, 1)
        dP = torch.matmul(C, dgrad).reshape(n_el, n_q, 2, 2)                   # (n_el, n_q, 2, 2)
        fe = torch.einsum("eqij,eqaj->eai", dP, dN_dX)                         # (n_el, nen, 2)
        Jv = torch.zeros(N, 2, dtype=v.dtype, device=v.device).index_add_(0, flat_conn, fe.reshape(-1, 2))
        return m * Jv.reshape(-1) + (1.0 - m) * v

    return A


def make_A_operator_qp_ew(u, theta, params):
    """
    The same stored-tangent product as ``make_A_operator_qp`` with every contraction written as an
    ELEMENT-WISE multiply followed by a sum (broadcasting), so that no batched GEMM (cuBLAS) is called.
    The matrices are tiny (2x2, 4x4, nen x 2), where a GEMM is dominated by tile padding and launch cost,
    whereas element-wise kernels are memory-bound.  Same result up to the order of summation (~1e-16).

        dgrad[e,q,i,j] = sum_a  ve[e,a,i] dN[e,q,a,j]
        dP[e,q,m]      = sum_n  C[e,q,m,n] dgrad[e,q,n]
        fe[e,a,i]      = sum_{q,j} dP[e,q,i,j] dN[e,q,a,j]
    """
    conn, m = params["conn"], params["m"]
    N = params["nodes"].shape[0]
    n_el = conn.shape[0]
    C, dN_dX = _qp_setup(u, theta, params)
    n_q = dN_dX.shape[1]
    flat_conn = conn.reshape(-1)

    def A(v):
        """Jv from the stored moduli, element-wise formulation; v has shape (2N,)."""
        ve = (m * v).reshape(N, 2)[conn]                                       # (n_el, nen, 2)
        dgrad = (ve[:, None, :, :, None] * dN_dX[:, :, :, None, :]).sum(2)     # (n_el, n_q, 2, 2)  [i, j]
        dP = (C * dgrad.reshape(n_el, n_q, 1, 4)).sum(-1).reshape(n_el, n_q, 2, 2)
        fe = (dP[:, :, None, :, :] * dN_dX[:, :, :, None, :]).sum(dim=(1, 4))  # (n_el, nen, 2)
        Jv = torch.zeros(N, 2, dtype=v.dtype, device=v.device).index_add_(0, flat_conn, fe.reshape(-1, 2))
        return m * Jv.reshape(-1) + (1.0 - m) * v

    return A


TANGENT_OPERATORS = ("jvp", "linearize", "qp", "qp_ew")


def make_tangent_operator(kind, u, theta, params):
    """Build the forward tangent operator by name: one of ``TANGENT_OPERATORS``."""
    if kind == "jvp":
        return make_A_operator(u, theta, params)
    if kind == "linearize":
        return make_A_operator_linearize(u, theta, params)
    if kind == "qp":
        return make_A_operator_qp(u, theta, params)
    if kind == "qp_ew":
        return make_A_operator_qp_ew(u, theta, params)
    raise ValueError(f"Unknown tangent operator '{kind}'. Choose one of {TANGENT_OPERATORS}.")


# =============================================================================
# 2. TRANSPOSE TANGENT OPERATOR: AT(v) = J^T(u) * v
# =============================================================================
def make_AT_operator(u, theta, params):
    """
    PURPOSE:
    --------
    Constructs the matrix-free transpose linear operator AT(v) = J^T(u) * v.

    WHY IS THIS NEEDED IF WE ALREADY HAVE make_A_operator?
    -------------------------------------------------------
    - In hyperelasticity, J is symmetric (J = J^T), so A = AT.
    - BUT in dissipative mechanics (plasticity with non-associated flow, damage, friction),
      J is NON-SYMMETRIC (J != J^T).
    - In Step 7 (Adjoint Sensitivity Analysis), the adjoint equation requires solving:
          J^T * lambda = d(Loss)/du.
    - Reverse-mode AD (torch.func.vjp) naturally computes J^T * v!
    """
    m = params["m"]
    
    # Setup reverse-mode AD graph around global_residual evaluated at state u
    _, vjp_fn = vjp(lambda w: global_residual(w, theta, params), u)
    
    def AT(v):
        """
        Matrix-free transpose operator: AT(v) = m * J^T(u)*(m*v) + (1 - m)*v
        """
        # vjp_fn computes the Vector-Jacobian Product (J^T * (m * v))
        (Jt_v,) = vjp_fn(m * v)
        
        # Apply the exact same symmetric Dirichlet identity pivoting
        return m * Jt_v + (1.0 - m) * v

    return AT


# =============================================================================
# 3. MATRIX-FREE JACOBI DIAGONAL CALCULATOR
# =============================================================================
def compute_jacobi_diagonal(u, theta, params):
    """
    PURPOSE:
    --------
    Computes the exact diagonal vector d_i = K_ii of the global tangent matrix
    WITHOUT ever assembling the global matrix K.

    ALGORITHM (Nested vmap):
    ------------------------
    Global diagonal = sum of element diagonals: K_ii = sum_e (L^e)^T K^e_kk
    Each element has 2*nen DOFs (8 for Q4, 12 for Tri6, 18 for Q9).
    1. Inner vmap: Evaluates the 8 diagonal entries of ONE element using local unit vectors.
    2. Outer vmap: Batches this across ALL elements in the mesh in parallel.
    3. Scatter-Add: Assembles element diagonals into the global diagonal vector.
    """
    element_connectivity = params["conn"]          # Shape: (num_elements, nodes_per_element)
    nodes = params["nodes"]                        # Shape: (num_nodes, 2)
    num_nodes = nodes.shape[0]
    num_elements, nodes_per_element = element_connectivity.shape
    dofs_per_element = 2 * nodes_per_element       # e.g. 8 for Q4, 18 for Q9

    # 1. Gather current displacements and coordinates for all elements
    nodal_displacements = u.reshape(num_nodes, 2)
    element_displacements = nodal_displacements[element_connectivity]  # (num_elements, nen, 2)
    element_p = element_material_parameters(theta, params)              # (num_elements, n_p)
    psi_fn = psi_2d(material_of(params)[0])

    # 2. Build local Cartesian unit basis vectors for ONE element
    # Why reshape to (-1, nodes_per_element, 2)?
    # Because `elem_residual` expects displacements of shape (nen, 2)!
    # torch.eye(2*nen) creates 2*nen vectors: e_0 = [1,0,...], e_1 = [0,1,...]
    # Reshaping to (8, 4, 2) formats each basis vector to match element displacement geometry:
    #   local_basis[0] = Node 0 moved +1 in X
    #   local_basis[1] = Node 0 moved +1 in Y
    #   ...
    #   local_basis[7] = Node 3 moved +1 in Y
    local_basis = torch.eye(
        dofs_per_element, dtype=u.dtype, device=u.device
    ).reshape(-1, nodes_per_element, 2)

    # 3. Action of an element tangent on unit direction e:
    #    e_k^T * K^e * e_k = K^e_kk (the k-th diagonal entry of the element tangent)
    def element_tangent_diag_entry(u_e, geom_e, p_e, e_vec):
        dN_dX_e, w_detJ_e = geom_e
        def res_w(w):
            return elem_residual_precomputed(w, dN_dX_e, w_detJ_e, p_e, psi_fn)
        # Directional derivative of element residual along local unit perturbation e_vec:
        _, Je_e = jvp(res_w, (u_e,), (e_vec,))
        # Dot product (Je_e · e_vec) extracts the scalar diagonal entry:
        return (Je_e * e_vec).sum()

    # 4. Inner vmap: loops over the 8 unit basis vectors for a SINGLE element
    # in_dims: element state is static (None); basis vector is batched (0).
    def single_element_diagonal(u_e, geom_e, p_e):
        return vmap(
            element_tangent_diag_entry,
            in_dims=(None, None, None, 0)
        )(u_e, geom_e, p_e, local_basis)

    # 5. Outer vmap: vectorizes across ALL elements in the mesh simultaneously!
    # Evaluates all elements in parallel across GPU threads:
    element_diagonals = vmap(
        single_element_diagonal,
        in_dims=(0, 0, 0)
    )(element_displacements, get_geometry(params), element_p)
    # Shape: (num_elements, 8) -> reshape to (num_elements, nen, 2)
    element_diagonals = element_diagonals.reshape(num_elements, nodes_per_element, 2)

    # 6. Global Assembly: Scatter-add element diagonals into global diagonal vector
    flat_connectivity = element_connectivity.reshape(-1)
    flat_diagonals = element_diagonals.reshape(-1, 2)

    diag_global = torch.zeros(num_nodes, 2, dtype=u.dtype, device=u.device)
    diag_global.index_add_(0, flat_connectivity, flat_diagonals)
    diag_flat = diag_global.reshape(-1)

    # 7. Dirichlet Boundary Pivot Handling:
    # Clamped DOFs have pivot value 1.0; Free DOFs are clamped to positive values
    diag_positive = torch.clamp(diag_flat, min=1e-6)
    return torch.where(params["m"] > 0.0, diag_positive, torch.ones_like(diag_flat))


# =============================================================================
# 4. PRECONDITIONER CLOSURE: M^-1(r) = r / diag
# =============================================================================
def make_jacobi_preconditioner(u, theta, params):
    """
    PURPOSE:
    --------
    Creates the callable preconditioner function M^-1(r) = r / diag(K).
    Handed to PCG in Step 6 to accelerate Krylov convergence.
    """
    diag = compute_jacobi_diagonal(u, theta, params)
    return lambda r: r / diag

# =============================================================================
# 5. STRONGER PRECONDITIONERS
# =============================================================================
# Jacobi uses one number per DOF.  The two preconditioners below capture more of the
# tangent and cut the number of CG iterations (each of which costs one jvp):
#
#   "jacobi"        r / diag(A)                     matrix-free, any device
#   "block_jacobi"  invert the 2x2 block of every   matrix-free, any device; couples
#                   node (u_x, u_y coupling)        the two displacement components
#   "ilu"           incomplete LU of the ASSEMBLED  needs the sparse matrix (CPU/SciPy);
#                   sparse tangent                  by far the fewest iterations
#
# All are closures  r -> M^-1 r  with the same interface, so ``pcg`` does not change.
PRECONDITIONERS = ("jacobi", "block_jacobi", "ilu")


def _element_tangent_blocks(u, theta, params):
    """
    The 2x2 diagonal blocks of every ELEMENT tangent, matrix-free:  (n_el, nen, 2, 2).

    For the local unit vector e_k (node a, component j) one jvp gives the whole column
    J^e e_k of shape (nen, 2); its row a is the block column  B[:, j]  of node a.
    """
    conn = params["conn"]
    N = params["nodes"].shape[0]
    nen = conn.shape[1]
    ue = u.reshape(N, 2)[conn]
    element_p = element_material_parameters(theta, params)
    psi_fn = psi_2d(material_of(params)[0])
    local_basis = torch.eye(2 * nen, dtype=u.dtype, device=u.device).reshape(-1, nen, 2)   # (2nen, nen, 2)
    node_of_dof = torch.arange(2 * nen, device=u.device) // 2                              # local node of dof k

    def column_block_entries(u_e, geom_e, p_e, e_vec, a):
        dN_dX_e, w_detJ_e = geom_e
        res = lambda w: elem_residual_precomputed(w, dN_dX_e, w_detJ_e, p_e, psi_fn)
        _, Je = jvp(res, (u_e,), (e_vec,))          # J^e e_k, shape (nen, 2)
        return Je[a]                                 # row of node a: (2,)  = B[:, j]

    def one_element(u_e, geom_e, p_e):
        cols = vmap(column_block_entries, in_dims=(None, None, None, 0, 0))(
            u_e, geom_e, p_e, local_basis, node_of_dof)          # (2nen, 2): [k=(a,j), i]
        return cols.reshape(nen, 2, 2).transpose(1, 2)           # [a, i, j]

    return vmap(one_element, in_dims=(0, 0, 0))(ue, get_geometry(params), element_p)   # (n_el, nen, 2, 2)


def make_block_jacobi_preconditioner(u, theta, params):
    """
    Nodal BLOCK-Jacobi:  M^-1 r  applies the inverse of the 2x2 diagonal block of
    every node.  Unlike point-Jacobi it couples u_x and u_y of a node, which matters
    for shear and bending.  Matrix-free: the blocks are assembled from element
    blocks with one scatter-add, exactly like the diagonal.

    Constrained DOFs get identity rows/columns (the same pivot as ``make_A_operator``).
    """
    conn, m = params["conn"], params["m"]
    N = params["nodes"].shape[0]
    blocks = _element_tangent_blocks(u, theta, params)                     # (n_el, nen, 2, 2)

    B = torch.zeros(N, 2, 2, dtype=u.dtype, device=u.device)
    B.index_add_(0, conn.reshape(-1), blocks.reshape(-1, 2, 2))            # scatter-add -> (N, 2, 2)

    mn = m.reshape(N, 2)                                                   # 1 free, 0 fixed
    B = B * (mn[:, :, None] * mn[:, None, :]) + torch.diag_embed(1.0 - mn)  # M B M + (I - M)
    B = B + 1e-12 * torch.eye(2, dtype=u.dtype, device=u.device)           # guard against singular blocks
    Binv = torch.linalg.inv(B)                                             # batched 2x2 inverses
    return lambda r: torch.einsum("nij,nj->ni", Binv, r.reshape(N, 2)).reshape(-1)


def assemble_tangent_sparse(u, theta, params):
    """
    Assemble the tangent as a SciPy sparse matrix (CSC), including the identity pivot
    on constrained DOFs.  ONLY used to build the ILU preconditioner -- the Krylov solver
    itself still applies the tangent matrix-free.

    Element matrices come from one forward-mode Jacobian per element
    (``jacfwd`` of the element force), then one COO scatter.
    """
    import numpy as np
    import scipy.sparse as sp
    from torch.func import jacfwd

    conn, m = params["conn"], params["m"]
    N = params["nodes"].shape[0]
    n_el, nen = conn.shape
    ue = u.reshape(N, 2)[conn]
    element_p = element_material_parameters(theta, params)
    psi_fn = psi_2d(material_of(params)[0])
    dN_dX, w_detJ = get_geometry(params)

    def element_matrix(u_e, dN_e, w_e, p_e):
        return jacfwd(lambda w: elem_residual_precomputed(w, dN_e, w_e, p_e, psi_fn))(u_e)   # (nen,2,nen,2)

    Ke = vmap(element_matrix)(ue, dN_dX, w_detJ, element_p).reshape(n_el, 2 * nen, 2 * nen)
    dofs = (2 * conn[:, :, None] + torch.arange(2, device=conn.device)).reshape(n_el, 2 * nen)   # (n_el, 2nen)
    rows = dofs[:, :, None].expand(-1, -1, 2 * nen).reshape(-1)
    cols = dofs[:, None, :].expand(-1, 2 * nen, -1).reshape(-1)
    free = m > 0.5
    vals = Ke.reshape(-1) * (free[rows] & free[cols])                     # drop rows/cols of constrained DOFs

    fixed_ids = torch.nonzero(~free).reshape(-1)
    rows = torch.cat([rows, fixed_ids])
    cols = torch.cat([cols, fixed_ids])
    vals = torch.cat([vals, torch.ones(fixed_ids.numel(), dtype=vals.dtype, device=vals.device)])   # identity pivot
    n = 2 * N
    return sp.coo_matrix((vals.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())), shape=(n, n)).tocsc()


def make_ilu_preconditioner(u, theta, params, drop_tol=1e-5, fill_factor=30.0):
    """
    Incomplete-LU preconditioner of the assembled tangent (SciPy ``spilu``).
    ``drop_tol`` discards small fill-in (smaller = better but costlier); ``fill_factor``
    caps the memory.  Measured on this code base: with ``drop_tol=1e-3`` the slender
    L-bracket still needed 120 CG iterations per solve, with ``1e-5`` only 2.
    At such a small tolerance ILU is an *approximate direct solver*: excellent up to
    a few 10^5 DOFs, but its memory grows faster than linearly -- for very large
    problems use ``block_jacobi``/``jacobi`` (matrix-free) or a multigrid method.  The factorisation runs on the CPU; the closure moves the residual
    there and back, so on a GPU this is only worthwhile when the iteration savings
    outweigh the transfers (i.e. for hard, ill-conditioned problems).
    """
    from scipy.sparse.linalg import spilu
    ilu = spilu(assemble_tangent_sparse(u, theta, params), drop_tol=drop_tol, fill_factor=fill_factor)

    def Minv(r):
        z = ilu.solve(r.detach().cpu().numpy())
        return torch.as_tensor(z, dtype=r.dtype, device=r.device)

    return Minv


def make_preconditioner(kind, u, theta, params):
    """Build a preconditioner by name: one of ``PRECONDITIONERS``."""
    if kind == "jacobi":
        return make_jacobi_preconditioner(u, theta, params)
    if kind == "block_jacobi":
        return make_block_jacobi_preconditioner(u, theta, params)
    if kind == "ilu":
        return make_ilu_preconditioner(u, theta, params)
    raise ValueError(f"Unknown preconditioner '{kind}'. Choose one of {PRECONDITIONERS}.")
