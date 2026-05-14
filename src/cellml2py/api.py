"""Public compile and simulation API.

This module exposes three user-facing functions:

* :func:`compile_cellml` - compile a raw CellML 1.0 / 1.1 file via the
  built-in XML parser and MathML translator.
* :func:`compile_opencor_python` - compile a Python file exported by the
  OpenCOR simulation environment.
* :func:`simulate` - integrate a compiled model with ``scipy.integrate.solve_ivp``.

Both compiler paths return an identical :class:`~cellml2py.contracts.CompiledModel`
so that downstream code is agnostic to the source format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import CompileOptions, CompiledModel
from .cellml_adapter import compile_cellml_model
from .opencor_adapter import compile_opencor_python_model


def compile_cellml(
    path: str | Path,
    backend: str = "numpy",
    options: CompileOptions | None = None,
) -> CompiledModel:
    """Compile a CellML 1.0 / 1.1 file into a runtime model.

    Runs the full direct-compilation pipeline:

    1. Optionally validates the document with ``libcellml`` if installed.
    2. Parses XML structure (components, variables, connections, equations).
    3. Translates Content MathML to NumPy expression strings.
    4. Builds and pre-compiles the NumPy RHS closure.

    Parameters
    ----------
    path:
        Path to the ``.cellml`` file (string or :class:`pathlib.Path`).
    backend:
        Numerical backend.  ``"numpy"`` (default) or ``"jax"``.
        The JAX backend requires ``pip install cellml2py[jax]`` and returns
        a model whose ``make_rhs()`` is safe for ``jax.jit``, ``jax.grad``,
        and ``diffrax.ODETerm``.
    options:
        Optional :class:`~cellml2py.contracts.CompileOptions` with override
        target declarations.  Defaults to no overrides.

    Returns
    -------
    CompiledModel
        Compiled model ready for simulation.
    """

    resolved_options = options or CompileOptions()
    return compile_cellml_model(Path(path), backend=backend, options=resolved_options)


def compile_opencor_python(
    path: str | Path,
    backend: str = "numpy",
    options: CompileOptions | None = None,
) -> CompiledModel:
    """Compile an OpenCOR-exported Python model into a runtime model.

    OpenCOR can export CellML models as self-contained Python files with
    ``initConsts``, ``computeRates``, and ``createLegends`` functions.  This
    function loads that file as a module, optionally patches ``computeRates``
    to honour override targets, and wraps the result in a standard
    :class:`~cellml2py.contracts.CompiledModel`.

    Parameters
    ----------
    path:
        Path to the OpenCOR-exported ``.py`` file.
    backend:
        Numerical backend.  ``"numpy"`` (default) or ``"jax"``.
        The JAX backend requires ``pip install cellml2py[jax]`` and returns
        a model whose ``make_rhs()`` is safe for ``jax.jit``, ``jax.grad``,
        and ``diffrax.ODETerm``.
    options:
        Optional :class:`~cellml2py.contracts.CompileOptions` with override
        target declarations.  Override targets may be algebraic variable names,
        constant names, or positional legend indices.

    Returns
    -------
    CompiledModel
        Compiled model ready for simulation.
    """

    resolved_options = options or CompileOptions()
    return compile_opencor_python_model(
        Path(path), backend=backend, options=resolved_options
    )


def simulate(
    model: CompiledModel,
    t_span: tuple[float, float],
    *,
    steps: int = 1001,
    t_eval: np.ndarray | None = None,
    forcing: Callable[[float], Sequence[float]] | None = None,
    params: dict[str, float] | None = None,
    method: str = "LSODA",
    max_step: float = float("inf"),
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """Integrate a compiled CellML model with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    model:
        Compiled model returned by :func:`compile_cellml` or
        :func:`compile_opencor_python`.
    t_span:
        ``(t0, t1)`` integration interval in the model's native time units.
    steps:
        Number of evenly-spaced output samples between *t0* and *t1*.
        Ignored when *t_eval* is provided explicitly.
    t_eval:
        Explicit 1-D array of output times.  When given, *steps* is ignored.
    forcing:
        ``forcing(t) -> sequence[float]`` — one value per override target in
        ``model.layout.forcing_names`` order.  Pass ``None`` (default) when no
        override targets were compiled in.
    params:
        Parameter overrides applied on every RHS call.  Keys must be canonical
        names as they appear in ``model.layout.parameter_names``.
    method:
        ODE solver method forwarded to ``solve_ivp``.  Stiff-capable choices:
        ``"LSODA"`` (default), ``"Radau"``, ``"BDF"``.
    max_step:
        Maximum allowed step size forwarded to ``solve_ivp``.
    rtol, atol:
        Relative and absolute tolerances forwarded to ``solve_ivp``.

    Returns
    -------
    scipy.integrate.OdeResult
        The solution bunch object from ``solve_ivp``.  Key attributes:
        ``.t`` (1-D time array), ``.y`` (state array of shape
        ``(n_states, n_t)``), ``.success``, ``.message``.
    """
    rhs = model.make_rhs()
    _params: dict[str, float] | None = params or None

    if forcing is None:

        def _fun(t: float, x: np.ndarray) -> np.ndarray:
            return rhs(t, x, _params)
    else:

        def _fun(t: float, x: np.ndarray) -> np.ndarray:
            return rhs(t, x, (_params, list(forcing(t))))

    _t_eval: np.ndarray = (
        t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], steps)
    )

    return solve_ivp(
        fun=_fun,
        t_span=t_span,
        y0=model.initial_state,
        t_eval=_t_eval,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )


def simulate_diffrax(
    model: CompiledModel,
    t_span: tuple[float, float],
    *,
    steps: int = 1001,
    dt0: float | None = None,
    forcing: Callable[[float], Sequence[float]] | None = None,
    params: dict[str, float] | None = None,
    solver: Any = None,
    stepsize_controller: Any = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 16**4,
    dtmin: float | None = None,
    jit: bool = False,
):
    """Integrate a compiled CellML model with ``diffrax``.

    Requires ``pip install cellml2py[jax]`` (installs JAX, jaxlib, and
    diffrax).  Designed for use with a ``backend="jax"`` compiled model so the
    RHS is fully JAX-traceable, but also accepts a ``"numpy"`` backend model
    (the RHS will then not be JIT-compilable).

    The RHS signature ``rhs(t, x, args)`` is already compatible with
    ``diffrax.ODETerm`` — no wrapper needed.

    Parameters
    ----------
    model:
        Compiled model returned by :func:`compile_cellml` or
        :func:`compile_opencor_python`.
    t_span:
        ``(t0, t1)`` integration interval in the model's native time units.
    steps:
        Number of evenly-spaced saved output times.
    dt0:
        Initial step size.  Defaults to ``(t1 - t0) / steps``.
    forcing:
        ``forcing(t) -> sequence[float]`` — one value per override target.
        Pass ``None`` when no override targets were compiled in.
    params:
        Parameter overrides applied on every RHS call.
    solver:
        A ``diffrax.AbstractSolver`` instance.  Defaults to
        ``diffrax.Kvaerno5()`` (an implicit, stiffly-stable solver suitable
        for electrophysiology models).
    stepsize_controller:
        A ``diffrax.AbstractStepSizeController``.  Defaults to
        ``diffrax.PIDController(rtol=rtol, atol=atol)``.
    rtol, atol:
        Tolerances forwarded to the default ``PIDController``.  Ignored when
        *stepsize_controller* is supplied explicitly.
    max_steps:
        Hard upper limit on the number of internal solver steps.
    dtmin:
        Minimum step size.
    jit:
        When ``True`` the entire ``diffrax.diffeqsolve`` call (integration
        loop + RHS) is compiled with ``jax.jit``.  Requires a
        ``backend="jax"`` model and a JAX-traceable *forcing* callable
        (no Python ``if``-branches on *t*).  The first call pays a
        one-time compilation cost; subsequent calls are significantly faster.

    Returns
    -------
    diffrax.Solution
        The diffrax solution object.  Key attributes:
        ``.ts`` (1-D time array of shape ``(steps,)``),
        ``.ys`` (state array of shape ``(steps, n_states)``),
        ``.result`` (``diffrax.RESULTS.successful`` on success).
    """
    try:
        import diffrax  # type: ignore
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "simulate_diffrax requires JAX and diffrax. "
            "Install with: pip install cellml2py[jax]"
        ) from exc

    rhs = model.make_rhs()
    if jit:
        rhs = jax.jit(rhs)
    _params: dict[str, float] | None = params or None

    if forcing is None:
        # No forcing — pass params directly as args.
        term = diffrax.ODETerm(rhs)
        _args = _params
        _use_args = True
    else:
        # Wrap RHS to inject forcing values at each call.  When jit=True,
        # forcing(t) must be JAX-traceable (no Python if-branches on t).
        _forcing = forcing

        def _rhs_with_forcing(t, x, _unused_args):
            fv = list(_forcing(t))
            return rhs(t, x, (_params, fv))

        term = diffrax.ODETerm(_rhs_with_forcing)
        _args = None
        _use_args = False

    t0, t1 = float(t_span[0]), float(t_span[1])
    _dt0 = float(dt0) if dt0 is not None else (t1 - t0) / steps
    _dtmin = float(dtmin) if dtmin is not None else None

    _solver = solver if solver is not None else diffrax.Kvaerno5()
    _controller = (
        stepsize_controller
        if stepsize_controller is not None
        else diffrax.PIDController(
            rtol=rtol,
            atol=atol,
            dtmin=_dtmin,
        )
    )

    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, steps))
    y0 = jnp.asarray(model.initial_state)
    _solve_args = _args if _use_args else None

    def _solve(y0):
        return diffrax.diffeqsolve(
            term,
            _solver,
            t0=t0,
            t1=t1,
            dt0=_dt0,
            y0=y0,
            args=_solve_args,
            saveat=saveat,
            stepsize_controller=_controller,
            max_steps=max_steps,
        )

    if jit:
        _solve = jax.jit(_solve)

    return _solve(y0)
