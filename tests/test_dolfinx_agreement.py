"""
Agreement with FEniCSx/dolfinx on an identical discrete problem (same mesh, energy, Gauss rule, loads).
Skipped unless DIFFFEA_DOLFINX_PYTHON points to a Python executable that has dolfinx installed (CI does
not have it); the full study is ``benchmarks/validate_against_dolfinx.py``.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

PYTHON = os.environ.get("DIFFFEA_DOLFINX_PYTHON")
pytestmark = pytest.mark.skipif(not PYTHON or not os.path.exists(PYTHON or ""),
                                reason="set DIFFFEA_DOLFINX_PYTHON to a python with dolfinx")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))


@pytest.mark.parametrize("geometry,element_type,h", [("plate_hole", "tri6", 0.4), ("plate_hole", "quad4", 0.4),
                                                      ("rectangle", "quad9", 0.5)])
def test_same_discrete_solution_as_dolfinx(geometry, element_type, h):
    import validate_against_dolfinx as v
    spec, params, u, fields, reaction, _ = v.solve_difffea(geometry, element_type, h)
    with tempfile.TemporaryDirectory() as work:
        ref = v.run_dolfinx(PYTHON, spec, params, work)
    assert np.abs(u.numpy() - ref["u"]).max() / np.abs(ref["u"]).max() < 1e-9
    assert abs(float(fields["strain_energy"]) - float(ref["energy"])) / abs(float(ref["energy"])) < 1e-9
