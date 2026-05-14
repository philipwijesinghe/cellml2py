"""Immutable data contracts shared between the compiler and the runtime.

All public types in this module are frozen dataclasses so that compiled models
are safe to share across threads and can be used as dictionary keys if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


OverrideKind = Literal["algebraic", "constant", "rate", "rate_addend"]


@dataclass(frozen=True)
class OverrideSpec:
    """Declares one runtime override (forcing) target.

    An override allows external code to replace or augment a variable's value
    on every RHS call by supplying a value through the forcing vector ``f``.
    This is the primary mechanism for injecting external stimuli (e.g. a
    current pulse) into a CellML model during simulation.

    Attributes
    ----------
    target:
        Variable name to override.  Accepts either a fully-qualified
        ``"<component>::<variable>"`` string or an unambiguous short name.
        For OpenCOR-exported models the special forms ``"algebraic[i]"``,
        ``"constant[i]"``, and ``"rate[i]"`` address legend indices directly.
    kind:
        How the override value is applied:

        * ``"algebraic"`` - replace the computed algebraic value entirely.
        * ``"constant"``  - replace the parameter/constant value.
        * ``"rate"``      - replace the computed ODE rate (d/dt).
        * ``"rate_addend"`` - add the forcing value to the computed rate,
          e.g. for injecting a stimulus current without replacing the full
          channel model.
        * ``None`` - kind is inferred automatically during compilation.
    """

    target: str
    kind: OverrideKind | None = None


@dataclass(frozen=True)
class CompileOptions:
    """Options controlling the compiler's behaviour.

    Attributes
    ----------
    override_targets:
        Sequence of :class:`OverrideSpec` instances that declare which
        variables should be driven by external forcing inputs.  The order
        determines the index mapping into the forcing vector supplied as the
        second element of the ``args`` tuple to ``rhs(t, x, (params, f))``.
    sanitize_nan:
        When ``True`` (default), any NaN or Inf values in the rate vector are
        replaced with ``0.0`` before the RHS returns.  This prevents adaptive
        ODE solvers from reducing their step size to machine epsilon when a
        model expression evaluates to NaN at a transient operating point.

        Set to ``False`` to preserve raw NaN/Inf values — useful when tracing
        the RHS with ``jax.grad`` for parameter fitting, where zeroing NaN
        would silently block gradient flow.
    """

    override_targets: tuple[OverrideSpec, ...] = ()
    sanitize_nan: bool = True


@dataclass(frozen=True)
class RuntimeLayout:
    """Deterministic variable ordering information exposed to callers.

    All name sequences are stable for the lifetime of a compiled model and
    define the index mapping between NumPy arrays and named variables.

    Attributes
    ----------
    state_names:
        Canonical variable names for each element of the state vector ``x``.
        ``x[i]`` corresponds to ``state_names[i]``.
    parameter_names:
        Canonical names for all constant parameters (those with an
        ``initial_value`` but no governing ODE).  Used as keys when
        overriding parameter values via the ``params`` argument.
    forcing_names:
        One name per override target, in the same order as the forcing vector.
        The forcing vector is the second element of the ``args`` tuple passed
        to ``rhs(t, x, (params, f))``.  Mirrors
        ``CompileOptions.override_targets``.
    override_targets:
        The :class:`OverrideSpec` objects as supplied to ``CompileOptions``.
    """

    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    forcing_names: tuple[str, ...]
    override_targets: tuple[OverrideSpec, ...]


@dataclass
class CompiledModel:
    """Compiled executable model and its associated metadata.

    Produced by :func:`~cellml2py.api.compile_cellml` or
    :func:`~cellml2py.api.compile_opencor_python`.  Holds all information
    needed to run a simulation without retaining a reference to the compiler.

    Attributes
    ----------
    backend:
        Name of the numerical backend used (currently always ``"numpy"``).
    layout:
        Variable ordering metadata; see :class:`RuntimeLayout`.
    initial_state:
        1-D NumPy array of initial state variable values sourced from the
        CellML ``initial_value`` attributes, in ``layout.state_names`` order.
    default_params:
        Dict of default parameter values keyed by both canonical and short
        names for convenience.
    """

    backend: str
    layout: RuntimeLayout
    initial_state: np.ndarray
    default_params: dict[str, float]
    _rhs_builder: Callable[[], Callable[..., Any]] = field(
        repr=False
    )

    @staticmethod
    def _unpack_args(
        args: Any,
    ) -> tuple[Any, Any]:
        """Unpack the ``args`` PyTree into ``(params, forcing)``.

        Accepts three shapes:

        * ``None``              — returns ``(None, None)``.
        * ``params``            — bare params dict; returns ``forcing = None``.
        * ``(params, forcing)`` — 2-tuple: params dict + forcing vector.

        This flexible contract makes ``rhs`` compatible with diffrax
        ``ODETerm``, ``jax.jit``, and plain SciPy usage under the same
        signature ``rhs(t, y, args)``.
        """
        if args is None:
            return None, None
        if isinstance(args, tuple) and len(args) == 2:
            return args[0], args[1]
        return args, None

    def make_rhs(self) -> Callable[..., Any]:
        """Construct the ``rhs(t, x, args)`` callable.

        The returned function has the signature::

            rhs(t, x, args) -> np.ndarray

        where ``args`` is unpacked via :meth:`_unpack_args` into
        ``(params, forcing)``:

        * ``rhs(t, x, None)``              — no overrides, no param changes.
        * ``rhs(t, x, {"g_K": 0.12})``     — param overrides only.
        * ``rhs(t, x, (params, f_t))``     — params + forcing vector.

        The JAX backend returns a diffrax-compatible callable; pass it
        directly to ``diffrax.ODETerm(model.make_rhs())`` without wrapping.
        """
        return self._rhs_builder()
