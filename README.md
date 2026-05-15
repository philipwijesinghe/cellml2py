# cellml2py

***Convert CellML models to a python callable r.h.s. function that computes rates of the ODE
for use with your favourite solver.***

> **Experimental.** This project was written for a specific use case.
> The use case worked, but the rest is not well supported and massaged into place with AI assistance.
> Expect rough edges and breaking bugs. 
> Contributions and changes are welcome.

The CellML standard hosts a large collection of biophisical dynamic models that
can are supported by third-party tools like OpenCOR. However, low-level interface
with these models for ODE integration can be useful for custom modelling, parameter
estimation, etc.

This package compiles CellML 1.x documents and OpenCOR-exported Python models
(using OpenCOR export to python utilities) into callable ODE RHS functions with
the signature `rhs(t, x, args) -> dxdt`, suitable for use with SciPy,
JAX/diffrax, or any ODE integrator.

```python
from cellml2py import compile_cellml

model = compile_cellml("data/opencor_models/hodgkin_huxley_squid_axon_model_1952.cellml")
rhs = model.make_rhs()
dxdt = rhs(0.0, model.initial_state, None)
```

## Features

- Direct CellML → NumPy or JAX compilation
- Support for OpenCOR-exported Python models (not well tested!)
- Forcing/override injection for replacing default constants, algebraics, or rates
- JAX backend for use with `jax.jit` and `diffrax.ODETerm`

## Install

```bash
pip install -e .           # core (NumPy)
pip install -e .[examples] # adds support for runnign examples/ .ipynb notebooks
pip install -e .[dev]      # adds pytest + scipy
pip install -e .[jax]      # adds JAX + diffrax
```

## Usage

**Compile and integrate a CellML file:**

```python
import numpy as np
from scipy.integrate import solve_ivp
from cellml2py import compile_cellml

model = compile_cellml("data/opencor_models/van_der_pol_model_1928.cellml")
rhs = model.make_rhs()

sol = solve_ivp(lambda t, x: rhs(t, x, None), (0.0, 10.0), model.initial_state)
```

**Override a model variable with an external signal (algebraic injection):**

```python
from cellml2py import compile_cellml, CompileOptions, OverrideSpec

opts = CompileOptions(override_targets=(OverrideSpec(target="membrane::i_Stim", kind="algebraic"),))
model = compile_cellml(
    "data/opencor_models/hodgkin_huxley_squid_axon_model_1952.cellml",
    options=opts,
)
rhs = model.make_rhs()

# args = (params, forcing_vector); pass None for params to use defaults
dxdt = rhs(0.0, model.initial_state, (None, [-20.0]))  # inject -20 µA/cm²
```

**Add an external drive to a state rate (rate addend):**

```python
from cellml2py import compile_cellml, CompileOptions, OverrideSpec

opts = CompileOptions(override_targets=(OverrideSpec(target="membrane::V", kind="rate_addend"),))
model = compile_cellml(
    "data/opencor_models/hodgkin_huxley_squid_axon_model_1952.cellml",
    options=opts,
)
rhs = model.make_rhs()

dxdt = rhs(0.0, model.initial_state, (None, [10.0]))  # add 10 mV/ms to dV/dt
```

See the notebooks in [examples/](examples/) for JAX/diffrax usage.

## Notes

Integration is as good as the numerical stability of the model. Many models can be very stiff and require careful choice of solver and tolerances. There are internal safe operations for exp, power, etc, but no guarantees that all models will be stable with the default settings. There is a default setting that replaces dxdt with zeros if they are nan or inf, but this will also break any gradient calcs.  See:

```python
options = CompileOptions(
    sanitize_nan=True,
)
```

## Tests

```bash
pytest
```

## Status

- Unit attributes are parsed but not validated or converted.
- Implicit algebraic equations (unknowns on both sides of `<eq>`) are not supported.
