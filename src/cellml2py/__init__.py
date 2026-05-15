"""cellml2py — compile CellML models to NumPy RHS callables.

Quick start::

    from cellml2py import compile_cellml, simulate

    model = compile_cellml("my_model.cellml")
    result = simulate(model, t_span=(0, 1000))

Two compiler paths are provided:

* :func:`compile_cellml` \u2013 direct XML / MathML compilation for CellML 1.0 / 1.1.
* :func:`compile_opencor_python` \u2013 wrap an OpenCOR-exported Python file.

Both return a :class:`CompiledModel` with a ``make_rhs()`` factory compatible
with ``scipy.integrate.solve_ivp``.  Use :class:`CompileOptions` and
:class:`OverrideSpec` to inject external forcing signals at runtime.

For stiff Hodgkin-Huxley models, use :func:`simulate_rush_larsen` (NumPy and
JAX backends) or :class:`~cellml2py.rl_solver.RushLarsenSolver` (diffrax) for
unconditionally stable gate-variable integration with fixed large time steps.
"""

from .api import compile_cellml, compile_opencor_python, simulate, simulate_diffrax, simulate_rush_larsen
from .contracts import CompiledModel, CompileOptions, OverrideSpec, RuntimeLayout
from .rl_solver import RushLarsenSolver

__all__ = [
    "compile_cellml",
    "compile_opencor_python",
    "simulate",
    "simulate_diffrax",
    "simulate_rush_larsen",
    "RushLarsenSolver",
    "CompiledModel",
    "CompileOptions",
    "OverrideSpec",
    "RuntimeLayout",
]
