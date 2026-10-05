"""
Interactive FEA workbench (Streamlit GUI)
=========================================

Choose a geometry, element type and element size, set supports and loads, and
solve the nonlinear (neo-Hookean) problem with the MATRIX-FREE Newton-Krylov
solver.  Every number on the screen comes from the functions in ``scr/``.

Run with
    streamlit run streamlit_app.py
"""

import io
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

# make the ``scr`` package importable when the app is started from any directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scr.Elements import element_names, get_element                       # noqa: E402
from scr.Materials import MATERIAL_LIBRARY, material_names, get_material            # noqa: E402
from scr.Gmsh_Mesh import GEOMETRIES, GMSH_AVAILABLE                      # noqa: E402
from scr.Problem import build_problem, default_spec, GEOMETRY_PRESETS     # noqa: E402
from scr.Operators import PRECONDITIONERS                                  # noqa: E402
from scr.Newton_Krylov import newton_krylov_solve                         # noqa: E402
from scr.Postprocess import compute_fields, support_reactions             # noqa: E402
from scr.Plotting import plot_field, plot_mesh, make_deformation_gif                            # noqa: E402
from scr.Boundary_Conditions import select_boundary_nodes                 # noqa: E402

SIDES = ["left", "right", "top", "bottom"]

# (label shown in the GUI) -> (key in compute_fields, colormap, unit/description)
FIELDS = {
    "Displacement magnitude |u|": ("u_mag_n", "viridis"),
    "Displacement u_x": ("ux_n", "coolwarm"),
    "Displacement u_y": ("uy_n", "coolwarm"),
    "von Mises stress": ("mises_n", "inferno"),
    "Cauchy stress sigma_xx": ("sxx_n", "coolwarm"),
    "Cauchy stress sigma_yy": ("syy_n", "coolwarm"),
    "Cauchy stress sigma_xy": ("sxy_n", "coolwarm"),
    "Volume ratio J = det F": ("J_n", "PuOr"),
    "Strain-energy density": ("energy_density_n", "magma"),
}


# =============================================================================
# Sidebar: collect a problem specification
# =============================================================================
def sidebar_spec() -> dict:
    """Draw all input widgets and return the problem specification dictionary."""
    sb = st.sidebar
    sb.title("Problem set-up")

    # ---- geometry ----------------------------------------------------------
    sb.subheader("1  Geometry")
    geometry = sb.selectbox("Shape", list(GEOMETRIES), format_func=lambda k: GEOMETRIES[k]["label"])
    geom_params = {}
    for name, default in GEOMETRIES[geometry]["params"].items():
        geom_params[name] = sb.number_input(f"{name}", value=float(default), min_value=0.05,
                                            step=0.1, format="%.2f", key=f"{geometry}_{name}")

    # ---- mesh --------------------------------------------------------------
    sb.subheader("2  Mesh")
    element_type = sb.selectbox("Element type", element_names(), index=0,
                                help="quad4/tri3: linear.  quad8 (serendipity), quad9, tri6: quadratic.")
    h = sb.slider("Element size h (smaller = denser mesh)", 0.04, 0.6, 0.2, 0.01)
    mesher_options = ["structured", "gmsh"] if geometry == "rectangle" else ["gmsh"]
    if not GMSH_AVAILABLE:
        mesher_options = ["structured"]
    mesher = sb.radio("Mesher", mesher_options, horizontal=True, key=f"mesher_{geometry}",
                      help="'structured' = simple grid generated in pure PyTorch (rectangle only).")

    # ---- material ----------------------------------------------------------
    sb.subheader("3  Material (hyperelastic, plane strain)")
    material = sb.selectbox("Constitutive model", material_names(),
                            format_func=lambda k: MATERIAL_LIBRARY[k]["label"])
    mat = get_material(material)
    material_params = {}
    cols = sb.columns(2)
    for i, (pname, default) in enumerate(zip(mat["param_names"], mat["defaults"])):
        material_params[pname] = cols[i % 2].number_input(pname, value=float(default), format="%.4g",
                                                          key=f"mat_{material}_{pname}")
    if "mu" in material_params and "lmbda" in material_params:        # equivalent engineering constants
        mu_, lm_ = material_params["mu"], material_params["lmbda"]
        sb.caption(f"initial small-strain moduli: E = {mu_ * (3 * lm_ + 2 * mu_) / (lm_ + mu_):.4g}, "
                   f"nu = {lm_ / (2 * (lm_ + mu_)):.3f}")
    if mat["fibrous"]:
        sb.caption("Fibres (angle beta to the x axis, +beta and -beta families) carry load only in tension; "
                   "d = dispersion (0 aligned, 1/3 isotropic); second_family 1 = two families, 0 = one.")

    # ---- supports and loads ------------------------------------------------
    preset = GEOMETRY_PRESETS[geometry]
    sb.subheader("4  Supports and load")
    fixed_side = sb.selectbox("Supported side", SIDES, index=SIDES.index(preset["fixed_side"]),
                              key=f"fs_{geometry}")
    fixed_dofs = sb.selectbox("Support type", ["xy", "x", "y"], index=0,
                              format_func={"xy": "clamped (u_x = u_y = 0)", "x": "roller: u_x = 0",
                                           "y": "roller: u_y = 0"}.get)
    load_side = sb.selectbox("Loaded side", SIDES, index=SIDES.index(preset["load_side"]),
                             key=f"ls_{geometry}")
    load_mode = sb.radio("Load type", ["force", "displacement"], horizontal=True,
                         help="force = total force spread uniformly along the side; "
                              "displacement = imposed movement of the side.")
    c1, c2 = sb.columns(2)
    default_vals = preset["load_value"] if load_mode == "force" else (0.2, 0.0)
    a = c1.number_input("x-component" if load_mode == "force" else "u_x", value=float(default_vals[0]),
                        step=10.0 if load_mode == "force" else 0.05, key=f"lx_{geometry}_{load_mode}")
    b = c2.number_input("y-component" if load_mode == "force" else "u_y", value=float(default_vals[1]),
                        step=10.0 if load_mode == "force" else 0.05, key=f"ly_{geometry}_{load_mode}")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = sb.selectbox("Compute device", devices, help="'cuda' appears when a GPU is available (e.g. on Google Colab).") if len(devices) > 1 else "cpu"
    return dict(device=device, geometry=geometry, geom_params=geom_params, element_type=element_type, h=h,
                mesher=mesher, material=material, material_params=material_params, fixed_side=fixed_side, fixed_dofs=fixed_dofs,
                load_side=load_side, load_mode=load_mode, load_value=(a, b), dtype=torch.float64)


def sidebar_solver() -> dict:
    """Solver controls."""
    sb = st.sidebar
    sb.subheader("5  Solver")
    n_steps = sb.slider("Load steps (continuation)", 1, 20, 3)
    max_newton = sb.slider("Max Newton iterations per step", 5, 60, 25)
    log_tol = sb.slider("Residual tolerance (log10, relative to |f|)", -12, -4, -8)
    precond = sb.selectbox("Krylov preconditioner", list(PRECONDITIONERS),
                           format_func={"jacobi": "Jacobi (matrix-free)",
                                        "block_jacobi": "Nodal block-Jacobi (matrix-free)",
                                        "ilu": "ILU of assembled tangent (fastest on CPU)"}.get,
                           index=2, help="ILU needs the sparse matrix and runs on the CPU; "
                                         "the two Jacobi variants are matrix-free and run on any device.")
    return dict(n_load_steps=n_steps, max_newton=max_newton, rel_tol=10.0 ** log_tol, preconditioner=precond)


# =============================================================================
# Computation
# =============================================================================
def build(spec: dict):
    """Mesh + boundary conditions -> params (returns (params, error message))."""
    try:
        return build_problem(spec), None
    except Exception as exc:                    # surface any meshing / set-up problem in the GUI
        return None, f"{type(exc).__name__}: {exc}"


def solve(params: dict, solver_opts: dict, callback=None) -> dict:
    """
    Run the Newton-Krylov solver and post-process; returns a results dictionary.
    ``callback(record)`` is called after every Newton iteration (live monitoring).
    """
    n_el = params["conn"].shape[0]
    theta = torch.ones(n_el, dtype=params["nodes"].dtype, device=params["nodes"].device)   # nominal material everywhere
    f_scale = max(1.0, float(torch.norm(params["f"])))               # size of the external load
    t0 = time.time()
    u, info = newton_krylov_solve(params, theta, n_load_steps=solver_opts["n_load_steps"],
                                  tol=solver_opts["rel_tol"] * f_scale,
                                  max_newton=solver_opts["max_newton"], verbose=False,
                                  callback=callback, preconditioner=solver_opts["preconditioner"])
    elapsed = time.time() - t0
    fields = compute_fields(u, theta, params) if torch.isfinite(u).all() else None
    reactions = support_reactions(u, theta, params) if fields is not None else None
    return dict(u=u, info=info, fields=fields, reactions=reactions, elapsed=elapsed,
                applied=params["f"].reshape(-1, 2).sum(0))


# =============================================================================
# Plots
# =============================================================================
def fig_mesh(params: dict, spec: dict) -> Figure:
    """Undeformed mesh with the supported and loaded nodes highlighted."""
    fig = Figure(figsize=(7, 5), dpi=110)
    ax = fig.subplots()
    nodes, conn = params["nodes"].cpu().numpy(), params["conn"].cpu().numpy()
    plot_mesh(ax, nodes, conn, params["element_type"], show_nodes=(conn.shape[0] < 400))
    fixed = select_boundary_nodes(params["nodes"], spec["fixed_side"]).cpu().numpy()
    loaded = select_boundary_nodes(params["nodes"], spec["load_side"]).cpu().numpy()
    ax.plot(nodes[fixed, 0], nodes[fixed, 1], "s", color="tab:blue", ms=4, label=f"supported ({spec['fixed_side']})")
    ax.plot(nodes[loaded, 0], nodes[loaded, 1], "^", color="tab:red", ms=4, label=f"loaded ({spec['load_side']})")
    # fibre directions of a fibrous material, drawn at the element centroids
    if get_material(spec["material"])["fibrous"]:
        pv = params["mat_p"]
        beta = np.deg2rad(float(pv[4]))
        second = float(pv[6]) > 0.5
        cen = nodes[conn][:, :get_element(params["element_type"])["n_corners"], :].mean(1)
        size = 0.35 * np.sqrt(np.ptp(nodes[:, 0]) * np.ptp(nodes[:, 1]) / max(1, conn.shape[0]))
        for sign in ((1.0, -1.0) if second else (1.0,)):
            d = np.array([np.cos(beta), sign * np.sin(beta)]) * size
            ax.add_collection(LineCollection(np.stack([cen - d, cen + d], axis=1),
                                             colors="tab:green", linewidths=1.2))
        ax.plot([], [], color="tab:green", lw=1.2, label="fibre direction")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def fig_field(params: dict, result: dict, field_label: str, scale: float, show_edges: bool) -> Figure:
    """Contour plot of one field on the deformed configuration."""
    key, cmap = FIELDS[field_label]
    values = result["fields"][key].cpu().numpy()
    fig = Figure(figsize=(8, 5.5), dpi=110)
    ax = fig.subplots()
    pc = plot_field(ax, params["nodes"].cpu().numpy(), params["conn"].cpu().numpy(), params["element_type"],
                    values, u=result["u"].cpu().numpy(), scale=scale, cmap=cmap, show_edges=show_edges)
    fig.colorbar(pc, ax=ax, label=field_label)
    ax.set_title(f"{field_label}   (deformation x{scale:g})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def fig_convergence(result: dict) -> Figure:
    """Residual history and Krylov iterations per Newton iteration."""
    hist = result["info"]["history"]
    fig = Figure(figsize=(8, 3.6), dpi=110)
    ax1, ax2 = fig.subplots(1, 2)
    if hist:
        res = [h["residual"] for h in hist]
        ax1.semilogy(range(1, len(res) + 1), res, "o-")
        # vertical lines at load-step boundaries
        steps = [h["step"] for h in hist]
        for i in range(1, len(steps)):
            if steps[i] != steps[i - 1]:
                ax1.axvline(i + 0.5, color="0.7", ls="--", lw=0.8)
        ax2.bar(range(1, len(hist) + 1), [h["cg_iters"] for h in hist], color="tab:green")
    ax1.set_xlabel("cumulative Newton iteration")
    ax1.set_ylabel("||R||")
    ax1.set_title("Nonlinear residual")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("cumulative Newton iteration")
    ax2.set_ylabel("CG iterations")
    ax2.set_title("Krylov (PCG) iterations")
    fig.tight_layout()
    return fig


# =============================================================================
# Live convergence monitor (like the job monitor of commercial FE codes)
# =============================================================================
def fig_live(history: list, tol: float) -> Figure:
    """Residual norm versus cumulative Newton iteration, drawn while the solver runs."""
    fig = Figure(figsize=(8, 3.4), dpi=100)
    ax = fig.subplots()
    res = [h["residual"] for h in history]
    ax.semilogy(range(1, len(res) + 1), res, "o-", color="tab:blue", ms=4)
    ax.axhline(tol, color="tab:red", ls="--", lw=1, label=f"tolerance {tol:.1e}")
    for i in range(1, len(history)):                       # mark load-step boundaries
        if history[i]["step"] != history[i - 1]["step"]:
            ax.axvline(i + 0.5, color="0.75", ls=":", lw=0.8)
    ax.set_xlabel("cumulative Newton iteration")
    ax.set_ylabel("residual norm ||R||")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def solve_with_live_monitor(params: dict, solver_opts: dict) -> dict:
    """
    Solve while updating a progress bar, a residual-vs-iteration plot and a text
    log after EVERY Newton iteration.
    """
    st.subheader("Solver monitor")
    bar = st.progress(0.0, text="starting...")
    plot_slot = st.empty()
    log_slot = st.empty()
    history, lines = [], []
    tol = solver_opts["rel_tol"] * max(1.0, float(torch.norm(params["f"])))

    def on_iteration(rec: dict):
        history.append(rec)
        bar.progress(min(1.0, rec["load_factor"]),
                     text=f"load factor {rec['load_factor']:.3f}  |  Newton iteration {rec['iteration'] + 1}"
                          f"  |  ||R|| = {rec['residual']:.3e}")
        lines.append(f"step {rec['step']:3d}  lf {rec['load_factor']:.3f}  iter {rec['iteration']:2d}  "
                     f"||R|| {rec['residual']:.4e}  CG {rec['cg_iters']:4d}  alpha {rec['alpha']:.3f}")
        plot_slot.pyplot(fig_live(history, tol), width="content")
        log_slot.code("\n".join(lines[-12:]), language="text")      # last 12 lines, like a .msg file

    result = solve(params, solver_opts, callback=on_iteration)
    ok = result["info"]["converged"]
    bar.progress(1.0 if ok else min(1.0, result["info"]["load_factor"]),
                 text="converged" if ok else "did NOT converge (see log)")
    return result


# =============================================================================
# Page
# =============================================================================
def main():
    st.set_page_config(page_title="Matrix-free FEA workbench", layout="wide")
    st.title("Matrix-free Newton-Krylov FEA in PyTorch")
    st.caption("Large-strain neo-Hookean elasticity | purely functional PyTorch (`torch.func`) | "
               "no global stiffness matrix is ever assembled")

    spec = sidebar_spec()
    solver_opts = sidebar_solver()
    run = st.sidebar.button("Solve", type="primary", width="stretch")

    # --- mesh (cheap: rebuilt on every widget change) ------------------------
    params, error = build(spec)
    if error:
        st.error(f"Could not build the problem: {error}")
        return

    n_nodes, n_el = params["nodes"].shape[0], params["conn"].shape[0]
    c = st.columns(5)
    c[0].metric("Elements", n_el)
    c[1].metric("Nodes", n_nodes)
    c[2].metric("DOFs", 2 * n_nodes)
    c[3].metric("Free DOFs", int(params["m"].sum()))
    c[4].metric("Gauss pts / element", params["GP"].shape[0])

    # --- solve on demand; keep the result in the session ---------------------
    # the result is valid only for the spec it was computed with
    key = repr({k: v for k, v in spec.items() if k != "dtype"}) + repr(solver_opts)
    if run:
        st.session_state["result"] = solve_with_live_monitor(params, solver_opts)
        st.session_state["result_key"] = key
    result = st.session_state.get("result") if st.session_state.get("result_key") == key else None

    # everything is shown on ONE page (no hidden tabs): results first, mesh last
    if result is not None and result["fields"] is not None:
        f = result["fields"]
        m = st.columns(5)
        m[0].metric("Converged", "yes" if result["info"]["converged"] else "NO")
        m[1].metric("max |u|", f"{float(f['u_mag'].max()):.4g}")
        m[2].metric("max von Mises", f"{float(f['mises'].max()):.4g}")
        m[3].metric("min det F", f"{float(f['min_J']):.4f}")
        m[4].metric("Solve time", f"{result['elapsed']:.1f} s")

    tab_res = st.container()
    tab_conv = st.container()
    tab_info = st.container()
    tab_mesh = st.container()
    tab_res.subheader("Results")
    tab_conv.subheader("Convergence")
    tab_info.subheader("Report")
    tab_mesh.subheader("Mesh")

    with tab_mesh:
        st.pyplot(fig_mesh(params, spec), width="content")
        st.caption(f"Element type **{spec['element_type']}** "
                   f"({get_element(spec['element_type'])['n_nodes']} nodes), "
                   f"mesher **{spec['mesher']}**, target size h = {spec['h']:.2f}.")

    if result is None:
        tab_res.info("Press **Solve** in the sidebar to run the analysis for the current set-up. "
                     "Results, convergence history and the report will appear here.")
        return

    info = result["info"]
    with tab_res:
        if result["fields"] is None:
            st.error("The solution contains NaN/Inf (an element inverted). Try more load steps or a smaller load.")
        else:
            if not info["converged"]:
                st.warning("Newton did not converge to the requested tolerance; showing the last iterate.")
            cc = st.columns([2, 1, 1])
            field = cc[0].selectbox("Field", list(FIELDS))
            scale = cc[1].number_input("Deformation scale", value=1.0, min_value=0.0, step=0.5)
            edges = cc[2].checkbox("Show element edges", value=True)
            st.pyplot(fig_field(params, result, field, scale, edges), width="content")

            # ---- animation of the loading history (GIF) ---------------------------
            st.subheader("Animation (GIF)")
            g1, g2 = st.columns([3, 1])
            gif_field = g1.selectbox("Field shown in the animation", list(FIELDS), key="gif_field")
            gif_key = (key, gif_field, scale)
            if g2.button("Regenerate GIF") or st.session_state.get("gif_key") != gif_key:
                with st.spinner("Rendering the animation..."):
                    fk, cm = FIELDS[gif_field]
                    st.session_state["gif"] = make_deformation_gif(
                        params, result["info"]["snapshots"], fk, gif_field, cm, scale)
                    st.session_state["gif_key"] = gif_key
            st.image(st.session_state["gif"])
            st.download_button("Download GIF", st.session_state["gif"], file_name="deformation.gif",
                               mime="image/gif")

    with tab_conv:
        st.pyplot(fig_convergence(result), width="content")
        if info["history"]:
            st.dataframe(pd.DataFrame(info["history"]), width="stretch", height=240)

    with tab_info:
        fields = result["fields"]
        st.write(f"**Converged:** {info['converged']}   |   Newton iterations: {info['total_newton']}   |   "
                 f"Krylov iterations: {info['total_cg']}   |   wall time: {result['elapsed']:.2f} s")
        if fields is not None:
            rx, ry = result["reactions"].tolist()
            ax_, ay_ = result["applied"].tolist()
            st.write("**Global equilibrium check**  (support reactions + applied loads = 0)")
            st.table(pd.DataFrame({"x": [rx, ax_, rx + ax_], "y": [ry, ay_, ry + ay_]},
                                  index=["support reaction", "applied force", "sum (should be ~0)"]))
            st.write(f"Strain energy: **{float(fields['strain_energy']):.4f}**   |   "
                     f"min det F: **{float(fields['min_J']):.4f}**   |   "
                     f"max |u|: **{float(fields['u_mag'].max()):.4f}**   |   "
                     f"max von Mises: **{float(fields['mises'].max()):.2f}**")
            # nodal results as CSV
            df = pd.DataFrame({"x": params["nodes"][:, 0], "y": params["nodes"][:, 1],
                               "ux": fields["ux"], "uy": fields["uy"], "u_mag": fields["u_mag"],
                               "von_mises": fields["mises_n"], "sxx": fields["sxx_n"],
                               "syy": fields["syy_n"], "sxy": fields["sxy_n"], "J": fields["J_n"]})
            st.download_button("Download nodal results (CSV)", df.to_csv(index=False),
                               file_name="fea_results.csv", mime="text/csv")


def launch_from_plain_python():
    """
    Called when this file is run with the normal "Run Python File" button (plain
    ``python streamlit_app.py``).  Streamlit apps must be started by Streamlit,
    so we start the Streamlit server for this very file and open the browser.
    """
    import threading
    import webbrowser
    from streamlit.web import cli as stcli
    port = 8501
    sys.argv = ["streamlit", "run", os.path.abspath(__file__), "--server.headless", "true",
                "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    sys.exit(stcli.main())


# Two ways to get here:
#   1. started by Streamlit (``streamlit run`` or the launcher above): draw the page
#   2. started with plain Python (VS Code's Run button): start Streamlit first
from streamlit.runtime import exists as _running_under_streamlit  # noqa: E402

if _running_under_streamlit():
    main()
elif __name__ == "__main__":
    launch_from_plain_python()
