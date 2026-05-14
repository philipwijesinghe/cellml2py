from pathlib import Path

import numpy as np
import pytest

from cellml2py import compile_cellml


def _discover_upstream_model_fixtures() -> tuple[Path, ...]:
    return tuple(sorted(Path("data/opencor_models").glob("*.cellml")))


UPSTREAM_MODELS = _discover_upstream_model_fixtures()


@pytest.fixture(scope="session")
def opencor_cellml_model_paths() -> tuple[Path, ...]:
    assert UPSTREAM_MODELS, "No CellML fixture models were found in data/opencor_models"
    return UPSTREAM_MODELS


@pytest.mark.parametrize("model_path", UPSTREAM_MODELS, ids=lambda path: path.stem)
def test_upstream_opencor_cellml_models_compile_and_run(model_path: Path):
    compiled = compile_cellml(model_path, backend="numpy")
    rhs = compiled.make_rhs()

    dx = rhs(0.0, compiled.initial_state, None)

    assert compiled.initial_state.ndim == 1
    assert dx.shape == compiled.initial_state.shape
    assert np.all(np.isfinite(dx))
    assert len(compiled.layout.state_names) == compiled.initial_state.size


@pytest.mark.parametrize("model_path", UPSTREAM_MODELS, ids=lambda path: path.stem)
def test_solve_ivp_forward_integration_produces_trajectory(model_path: Path):
    scipy = pytest.importorskip("scipy.integrate")

    compiled = compile_cellml(model_path, backend="numpy")
    rhs = compiled.make_rhs()

    solution = scipy.solve_ivp(
        fun=lambda t, x: rhs(t, x, None),
        t_span=(0.0, 2.0),
        y0=compiled.initial_state,
        t_eval=np.linspace(0.0, 2.0, 41),
        max_step=0.05,
    )

    assert solution.success
    assert solution.y.shape == (compiled.initial_state.size, 41)
    assert np.all(np.isfinite(solution.y))
    assert not np.allclose(solution.y[:, 0], solution.y[:, -1])


def test_jax_backend_rhs_matches_numpy_at_initial_state():
    """JAX and NumPy backends produce identical rates at the initial state."""
    jnp = pytest.importorskip("jax.numpy")
    model_path = Path("data/opencor_models/van_der_pol_model_1928.cellml")

    np_model = compile_cellml(model_path, backend="numpy")
    jax_model = compile_cellml(model_path, backend="jax")

    rhs_np = np_model.make_rhs()
    rhs_jax = jax_model.make_rhs()

    x0 = np_model.initial_state
    dx_np = rhs_np(0.0, x0, None)
    dx_jax = rhs_jax(0.0, jnp.asarray(x0), None)

    np.testing.assert_allclose(np.asarray(dx_jax), dx_np, rtol=1e-5)


def test_jax_jit_compiles_and_runs():
    """The JAX backend RHS survives ``jax.jit`` without tracer errors."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    model_path = Path("data/opencor_models/van_der_pol_model_1928.cellml")

    jax_model = compile_cellml(model_path, backend="jax")
    rhs = jax_model.make_rhs()
    x0 = jnp.asarray(jax_model.initial_state)

    jitted = jax.jit(rhs)
    result = jitted(0.0, x0, None)

    assert result.shape == x0.shape
    assert jnp.all(jnp.isfinite(result))


def test_hh_rhs_depends_on_state_values():
    model_path = Path("data/opencor_models/hodgkin_huxley_squid_axon_model_1952.cellml")
    compiled = compile_cellml(model_path, backend="numpy")
    rhs = compiled.make_rhs()

    x0 = compiled.initial_state.copy()
    x1 = x0.copy()
    x1[0] = x1[0] + 50.0

    dx0 = rhs(0.0, x0, None)
    dx1 = rhs(0.0, x1, None)

    assert not np.allclose(dx0, dx1)