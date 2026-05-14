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
"""

from .api import compile_cellml, compile_opencor_python, simulate, simulate_diffrax
from .contracts import CompiledModel, CompileOptions, OverrideSpec, RuntimeLayout

__all__ = [
    "compile_cellml",
    "compile_opencor_python",
    "simulate",
    "simulate_diffrax",
    "CompiledModel",
    "CompileOptions",
    "OverrideSpec",
    "RuntimeLayout",
]
