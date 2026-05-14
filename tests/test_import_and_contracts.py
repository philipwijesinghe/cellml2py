from pathlib import Path

import numpy as np

from cellml2py import CompileOptions, OverrideSpec, compile_cellml, compile_opencor_python


def test_compile_and_contract_shape():
    model_path = Path("data/fabbri_SAN_2017.py")
    compiled = compile_opencor_python(model_path, backend="numpy")
    rhs = compiled.make_rhs()

    dx = rhs(0.0, compiled.initial_state, None)
    assert isinstance(dx, np.ndarray)
    assert dx.shape == compiled.initial_state.shape
    assert len(compiled.layout.state_names) == compiled.initial_state.size


def test_override_layout_names():
    model_path = Path("data/fabbri_SAN_2017.py")
    options = CompileOptions(override_targets=(OverrideSpec(target="i_tot"),))
    compiled = compile_opencor_python(model_path, backend="numpy", options=options)

    assert compiled.layout.forcing_names == ("i_tot",)


def test_compile_cellml_reads_sample_and_compiles():
    model_path = Path("data/HumanSAN_Fabbri_Fantini_Wilders_Severi_2017.cellml")
    compiled = compile_cellml(model_path, backend="numpy")
    rhs = compiled.make_rhs()
    ref = compile_opencor_python(Path("data/fabbri_SAN_2017.py"), backend="numpy")
    ref_rhs = ref.make_rhs()

    state_by_name = dict(zip(compiled.layout.state_names, compiled.initial_state))
    ref_state = np.asarray([state_by_name[name] for name in ref.layout.state_names], dtype=float)

    dx = rhs(0.0, compiled.initial_state, None)
    ref_dx = ref_rhs(0.0, ref_state, None)
    assert isinstance(dx, np.ndarray)
    assert dx.shape == compiled.initial_state.shape
    np.testing.assert_allclose(dx, ref_dx, rtol=1e-2, atol=1e-8)


def test_sanitize_nan_false_preserves_valid_array():
    """With sanitize_nan=False, finite rates are unchanged (no spurious zeroing)."""
    from cellml2py.contracts import CompileOptions

    model_path = Path("data/opencor_models/van_der_pol_model_1928.cellml")
    opts_true = CompileOptions(sanitize_nan=True)
    opts_false = CompileOptions(sanitize_nan=False)

    m_true = compile_cellml(model_path, backend="numpy", options=opts_true)
    m_false = compile_cellml(model_path, backend="numpy", options=opts_false)

    dx_true = m_true.make_rhs()(0.0, m_true.initial_state, None)
    dx_false = m_false.make_rhs()(0.0, m_false.initial_state, None)

    # For a model with no NaN in the rates, both should agree exactly.
    np.testing.assert_array_equal(dx_true, dx_false)

