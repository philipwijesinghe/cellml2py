from pathlib import Path

import numpy as np
import pytest

from cellml2py import CompileOptions, OverrideSpec, compile_opencor_python
from cellml2py.exceptions import ShapeError


def test_constant_override_via_forcing_changes_rates():
    model_path = Path("data/fabbri_SAN_2017.py")
    options = CompileOptions(override_targets=(OverrideSpec(target="ACh", kind="constant"),))
    compiled = compile_opencor_python(model_path, backend="numpy", options=options)
    rhs = compiled.make_rhs()

    baseline = rhs(0.0, compiled.initial_state, ({}, [0.0]))
    changed = rhs(0.0, compiled.initial_state, ({}, [0.5]))

    assert not np.allclose(baseline, changed)


def test_algebraic_override_i_tot_controls_voltage_rate():
    model_path = Path("data/fabbri_SAN_2017.py")
    options = CompileOptions(override_targets=(OverrideSpec(target="i_tot", kind="algebraic"),))
    compiled = compile_opencor_python(model_path, backend="numpy", options=options)
    rhs = compiled.make_rhs()

    forced = rhs(0.0, compiled.initial_state, ({}, [0.0]))
    assert abs(forced[0]) < 1e-14


def test_forcing_length_validation():
    model_path = Path("data/fabbri_SAN_2017.py")
    options = CompileOptions(override_targets=(OverrideSpec(target="i_tot", kind="algebraic"),))
    compiled = compile_opencor_python(model_path, backend="numpy", options=options)
    rhs = compiled.make_rhs()

    with pytest.raises(ShapeError):
        rhs(0.0, compiled.initial_state, ({}, []))
