from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from cellml2py import compile_opencor_python


def _load_module(path: Path):
    spec = spec_from_file_location("_fabbri_ref", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compiled_rhs_matches_exported_compute_rates_baseline():
    model_path = Path("data/fabbri_SAN_2017.py")
    compiled = compile_opencor_python(model_path, backend="numpy")
    rhs = compiled.make_rhs()

    ref_mod = _load_module(model_path)
    x0, c0 = ref_mod.initConsts()
    expected = np.asarray(ref_mod.computeRates(0.0, x0, c0), dtype=float)

    got = rhs(0.0, compiled.initial_state, None)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_opencor_jax_backend_rhs_matches_numpy():
    """OpenCOR JAX backend produces same rates as NumPy backend at initial state."""
    jnp = pytest.importorskip("jax.numpy")
    model_path = Path("data/fabbri_SAN_2017.py")

    np_model = compile_opencor_python(model_path, backend="numpy")
    jax_model = compile_opencor_python(model_path, backend="jax")

    rhs_np = np_model.make_rhs()
    rhs_jax = jax_model.make_rhs()

    x0 = np_model.initial_state
    dx_np = rhs_np(0.0, x0, None)
    dx_jax = rhs_jax(0.0, jnp.asarray(x0), None)

    np.testing.assert_allclose(np.asarray(dx_jax), dx_np, rtol=1e-3, atol=1e-8)
