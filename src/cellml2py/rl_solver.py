"""Rush-Larsen diffrax solver for cellml2py.

Provides :class:`RushLarsenSolver`, an :class:`~diffrax.AbstractSolver` that
wraps the :meth:`~cellml2py.CompiledModel.make_stepper` callable produced by
:func:`~cellml2py.compile_cellml` when gate variables are detected.

Usage
-----
.. code-block:: python

    import diffrax
    from cellml2py import compile_cellml
    from cellml2py.rl_solver import RushLarsenSolver

    model = compile_cellml("my_model.cellml", backend="jax")
    solver = RushLarsenSolver(model.make_stepper())

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model.make_rhs()),
        solver,
        t0=0.0, t1=4000.0,
        dt0=0.1,
        y0=jnp.asarray(model.initial_state),
        args=None,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 4000.0, 40001)),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

The solver bypasses ``ODETerm.vf`` for state updates — it calls the compiled
stepper directly.  Pass a real ``ODETerm(rhs)`` for the ``terms`` argument;
diffrax requires it for the ``func`` delegate but it is never used to advance
the state.
"""
from __future__ import annotations

from typing import Any

try:
    import equinox as eqx
    import diffrax
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RushLarsenSolver requires diffrax and equinox. "
        "Install them with: pip install diffrax equinox"
    ) from exc


class RushLarsenSolver(diffrax.AbstractSolver):  # type: ignore[misc]
    """Fixed-step exponential integrator for HH-style gating ODEs.

    HH gate variables satisfy ``dy/dt = (y_inf(V) - y) / tau(V)``.  The
    exact solution for a locally constant ``(y_inf, tau)`` is:

    .. math::

        y(t + \\Delta t) = y_\\infty + (y(t) - y_\\infty)\\,
            \\mathrm{e}^{-\\Delta t / \\tau}

    This update is unconditionally stable regardless of how small ``tau``
    becomes, removing the stiffness contribution of the gating subsystem.

    All other state variables are advanced with forward Euler.

    Parameters
    ----------
    stepper_fn:
        The callable returned by :meth:`~cellml2py.CompiledModel.make_stepper`.
        It has the signature ``step(t, x, dt, args) -> x_new`` and is fully
        ``jax.jit``-compilable.

    Notes
    -----
    Use ``diffrax.ConstantStepSize()`` as the step-size controller; this solver
    does not produce an error estimate so adaptive controllers will fail.
    """

    stepper_fn: Any = eqx.field(static=True)
    interpolation_cls: Any = eqx.field(
        default=diffrax.LocalLinearInterpolation, static=True
    )

    @property
    def term_structure(self):  # type: ignore[override]
        return diffrax.ODETerm

    def order(self, terms: Any) -> int:  # type: ignore[override]
        return 1

    def init(  # type: ignore[override]
        self,
        terms: Any,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
    ) -> None:
        return None

    def step(  # type: ignore[override]
        self,
        terms: Any,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        solver_state: Any,
        made_jump: Any,
    ) -> tuple[Any, None, dict, None, Any]:
        """Advance *y0* from *t0* to *t1* using the Rush-Larsen stepper."""
        dt = t1 - t0
        y1 = self.stepper_fn(t0, y0, dt, args)
        # dense_info must contain y0 and y1 for LocalLinearInterpolation
        # (required by diffrax ≥ 0.7.x).  No error estimate — caller must
        # use ConstantStepSize().
        return y1, None, dict(y0=y0, y1=y1), None, diffrax.RESULTS.successful

    def func(  # type: ignore[override]
        self,
        terms: Any,
        t0: Any,
        y0: Any,
        args: Any,
    ) -> Any:
        """Delegate to *terms.vf* for derivative evaluation (used by dense output)."""
        return terms.vf(t0, y0, args)
