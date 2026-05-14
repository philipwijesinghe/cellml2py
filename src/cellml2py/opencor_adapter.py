"""OpenCOR-exported Python model compiler path.

OpenCOR (https://opencor.ws) can export CellML models as self-contained Python
files.  These files expose three functions:

* ``initConsts()`` – returns ``(states, constants)`` lists with initial values.
* ``computeRates(voi, states, rates, algebraic, constants)`` – fills the
  ``rates`` array in-place for a given state point.
* ``createLegends()`` – returns ``(legend_states, legend_algebraic,
  legend_voi, legend_constants)`` with human-readable variable descriptions.

This module loads such a file as a Python module, extracts variable ordering
from the legends, optionally patches ``computeRates`` via AST rewriting to
honour :class:`~cellml2py.contracts.OverrideSpec` targets, and wraps the
result in the standard :class:`~cellml2py.contracts.CompiledModel` contract.

AST patching strategy
---------------------
When override targets are present, ``_build_overridden_compute_rates`` parses
the original source of the exported file, deep-copies the ``computeRates``
function node, and inserts a helper ``_ovr(idx, default)`` that returns the
forcing value for index *idx* if one has been registered, or the original
default otherwise.  Every ``algebraic[i] = expr`` assignment is then wrapped
as ``algebraic[i] = _ovr(i, expr)``.  The patched function is compiled to
bytecode and injected into the module's namespace.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .contracts import CompileOptions, CompiledModel, OverrideKind, OverrideSpec, RuntimeLayout
from .exceptions import MissingInputError, ShapeError, UnsupportedFeatureError


@dataclass(frozen=True)
class _ResolvedOverride:
    """Internal: a fully-resolved override target with its array index.

    Attributes
    ----------
    kind:
        The resolved override kind (``"algebraic"``, ``"constant"``, or
        ``"rate"``).
    index:
        Zero-based index into the corresponding OpenCOR legend array.
    spec:
        The original :class:`~cellml2py.contracts.OverrideSpec` that produced
        this resolution.
    """

    kind: OverrideKind
    index: int
    spec: OverrideSpec


class _AlgebraicOverrideTransformer(ast.NodeTransformer):
    """AST transformer that wraps every ``algebraic[i] = expr`` assignment.

    For each assignment whose target is ``algebraic[<int>]``, the RHS
    expression ``expr`` is replaced with ``_ovr(<int>, expr)``.  At runtime
    the injected ``_ovr`` helper returns the forcing value if that index was
    registered as an override target, or the original expression value if not.
    This allows selective algebraic overrides without rewriting the full
    ``computeRates`` body.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if len(node.targets) != 1:
            return node

        idx = _extract_list_index(node.targets[0], "algebraic")
        if idx is None:
            return node

        node.value = ast.Call(
            func=ast.Name(id="_ovr", ctx=ast.Load()),
            args=[ast.Constant(value=idx), node.value],
            keywords=[],
        )
        return node


def _extract_list_index(node: ast.AST, list_name: str) -> int | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != list_name:
        return None

    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
        return slice_node.value
    if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
        value = slice_node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return value.value
    return None


def _load_module_from_path(path: Path) -> ModuleType:
    """Dynamically import an OpenCOR-exported Python file as a module.

    A unique synthetic module name is derived from the file's absolute path
    hash to avoid collisions when multiple models are loaded in the same
    process.  The module is *not* added to ``sys.modules`` so it will not
    interfere with other imports.

    Raises
    ------
    UnsupportedFeatureError
        If ``importlib`` cannot create a module spec for *path*.
    """
    module_name = f"_cellml2py_opencor_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise UnsupportedFeatureError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legend_parts(legend: str) -> tuple[str, str] | None:
    """Parse an OpenCOR legend string into ``(component, variable)``.

    OpenCOR legend strings follow the format::

        "<variable> in component <component> (<units>)"

    Returns ``None`` if the string does not match this pattern.
    """
    if not legend or " in component " not in legend:
        return None
    lhs, rhs = legend.split(" in component ", 1)
    component = rhs.split(" (", 1)[0].strip()
    var = lhs.strip()
    if not var or not component:
        return None
    return component, var


def _build_names(legends: list[str], prefix: str) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Build qualified name lists and look-up dicts from an OpenCOR legend list.

    Parameters
    ----------
    legends:
        List of legend strings from ``createLegends()``.
    prefix:
        Fallback prefix used when a legend entry cannot be parsed
        (e.g. ``"algebraic"`` produces ``"algebraic[i]"`` as the name).

    Returns
    -------
    tuple[list[str], dict[str, int], dict[str, int]]
        ``(qualified_names, q_to_idx, short_unique_to_idx)`` where:

        * *qualified_names* – ``"<component>::<variable>"`` strings in legend order.
        * *q_to_idx* – maps each qualified name to its legend index.
        * *short_unique_to_idx* – maps each short name to its legend index,
          only for names that appear exactly once (to avoid ambiguous look-ups).
    """
    qualified: list[str] = []
    short_counts: dict[str, int] = {}

    for i, legend in enumerate(legends):
        parts = _legend_parts(legend)
        if parts is None:
            q = f"{prefix}[{i}]"
            short = q
        else:
            component, var = parts
            q = f"{component}::{var}"
            short = var
        qualified.append(q)
        short_counts[short] = short_counts.get(short, 0) + 1

    q_to_idx: dict[str, int] = {}
    short_unique_to_idx: dict[str, int] = {}
    for i, q in enumerate(qualified):
        q_to_idx[q] = i
        short = q.split("::", 1)[-1]
        if short_counts.get(short, 0) == 1:
            short_unique_to_idx[short] = i

    return qualified, q_to_idx, short_unique_to_idx


def _resolve_override_target(
    spec: OverrideSpec,
    algebraic_q_to_idx: dict[str, int],
    algebraic_short_to_idx: dict[str, int],
    constant_q_to_idx: dict[str, int],
    constant_short_to_idx: dict[str, int],
    n_rates: int,
) -> _ResolvedOverride:
    """Map an :class:`~cellml2py.contracts.OverrideSpec` to a concrete legend index.

    Resolution order:

    1. Positional forms ``"algebraic[i]"``, ``"constant[i]"``, ``"rate[i]"``.
    2. Qualified name ``"<component>::<variable>"`` against the appropriate
       look-up dict, respecting any explicit *kind* hint on the spec.
    3. Short (unqualified) name against the unique-only short-name look-up.
    4. Pure integer string for rate overrides.

    Raises
    ------
    MissingInputError
        If the target cannot be resolved through any of the above strategies.
    """
    target = spec.target.strip()
    kind = spec.kind

    if target.startswith("algebraic[") and target.endswith("]"):
        idx = int(target[len("algebraic[") : -1])
        return _ResolvedOverride(kind="algebraic", index=idx, spec=spec)
    if target.startswith("constant[") and target.endswith("]"):
        idx = int(target[len("constant[") : -1])
        return _ResolvedOverride(kind="constant", index=idx, spec=spec)
    if target.startswith("rate[") and target.endswith("]"):
        idx = int(target[len("rate[") : -1])
        return _ResolvedOverride(kind="rate", index=idx, spec=spec)

    if kind == "algebraic" or kind is None:
        if target in algebraic_q_to_idx:
            return _ResolvedOverride(kind="algebraic", index=algebraic_q_to_idx[target], spec=spec)
        if target in algebraic_short_to_idx:
            return _ResolvedOverride(kind="algebraic", index=algebraic_short_to_idx[target], spec=spec)

    if kind == "constant" or kind is None:
        if target in constant_q_to_idx:
            return _ResolvedOverride(kind="constant", index=constant_q_to_idx[target], spec=spec)
        if target in constant_short_to_idx:
            return _ResolvedOverride(kind="constant", index=constant_short_to_idx[target], spec=spec)

    if kind == "rate" and target.isdigit():
        idx = int(target)
        if idx < 0 or idx >= n_rates:
            raise MissingInputError(f"Invalid rate override index: {idx}")
        return _ResolvedOverride(kind="rate", index=idx, spec=spec)

    raise MissingInputError(
        f"Could not resolve override target '{spec.target}'. "
        "Use algebraic[index], constant[index], rate[index], or known legend names."
    )


def _build_overridden_compute_rates(module: ModuleType, source_path: Path):
    """Patch ``computeRates`` in *module* to honour algebraic overrides.

    Reads the original source from *source_path*, parses the AST, deep-copies
    the ``computeRates`` function node, renames it to
    ``_computeRates_overridden``, appends ``forcing`` and
    ``override_index_map`` parameters, applies
    :class:`_AlgebraicOverrideTransformer`, and compiles + exec-s the result
    into *module*'s ``__dict__``.

    Returns the injected ``_computeRates_overridden`` callable.
    """
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    compute_rates_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "computeRates":
            compute_rates_node = node
            break

    if compute_rates_node is None:
        raise UnsupportedFeatureError("OpenCOR module does not define computeRates")

    new_fn = copy.deepcopy(compute_rates_node)
    new_fn.name = "_computeRates_overridden"
    new_fn.args.args.append(ast.arg(arg="forcing"))
    new_fn.args.args.append(ast.arg(arg="override_index_map"))

    helper_src = (
        "def _ovr(_idx, _value):\n"
        "    _fi = override_index_map.get(_idx)\n"
        "    if _fi is None:\n"
        "        return _value\n"
        "    return forcing[_fi]\n"
    )
    helper_nodes = ast.parse(helper_src).body

    transformed = _AlgebraicOverrideTransformer().visit(new_fn)
    assert isinstance(transformed, ast.FunctionDef)
    transformed.body = helper_nodes + transformed.body

    mod = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, str(source_path), "exec")
    exec(code, module.__dict__)
    return module.__dict__["_computeRates_overridden"]


class _JaxComputeRatesTransformer(ast.NodeTransformer):
    """AST transformer that rewrites ``computeRates`` for JAX tracing.

    OpenCOR exports ``computeRates(voi, states, constants)`` which creates
    ``rates`` and ``algebraic`` as Python lists internally (e.g.
    ``rates = [0.0] * sizeStates``), mutates them with
    ``rates[i] = expr``, and returns ``rates``.  This is incompatible with
    JAX's functional array model.

    The transformer:

    1. Renames the function to ``_computeRates_jax``.
    2. Rewrites ``arr = [0.0] * <name>`` initialisations for ``rates`` and
       ``algebraic`` to ``arr = jnp.zeros(<name>)`` so that downstream
       item-set calls produce JAX arrays.
    3. Rewrites every ``arr[i] = expr`` assignment targeting ``rates`` or
       ``algebraic`` to ``arr = arr.at[i].set(expr)``.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "_computeRates_jax"
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Process children first.
        node = self.generic_visit(node)
        if len(node.targets) != 1:
            return node

        target = node.targets[0]

        # Rewrite initialisation: ``arr = [0.0] * size_name``  →
        #                         ``arr = jnp.zeros(size_name)``
        if (
            isinstance(target, ast.Name)
            and target.id in ("rates", "algebraic")
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Mult)
        ):
            # [0.0] * sizeXxx  — the list operand can be on either side.
            lhs, rhs_node = node.value.left, node.value.right
            if isinstance(lhs, ast.List) or isinstance(rhs_node, ast.List):
                size_node = rhs_node if isinstance(lhs, ast.List) else lhs
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="zeros",
                        ctx=ast.Load(),
                    ),
                    args=[size_node],
                    keywords=[],
                )
                return ast.Assign(
                    targets=[target],
                    value=new_value,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )

        # Rewrite: arr[i] = expr  →  arr = arr.at[i].set(expr)
        for arr_name in ("rates", "algebraic"):
            idx = _extract_list_index(target, arr_name)
            if idx is not None:
                at_set = ast.Assign(
                    targets=[ast.Name(id=arr_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Subscript(
                                value=ast.Attribute(
                                    value=ast.Name(id=arr_name, ctx=ast.Load()),
                                    attr="at",
                                    ctx=ast.Load(),
                                ),
                                slice=ast.Constant(value=idx),
                                ctx=ast.Load(),
                            ),
                            attr="set",
                            ctx=ast.Load(),
                        ),
                        args=[node.value],
                        keywords=[],
                    ),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                return at_set
        return node


def _build_jax_compute_rates(module: Any, source_path: Path):
    """Build a JAX-traceable ``_computeRates_jax`` callable from an OpenCOR module.

    Parses the source, deep-copies ``computeRates``, applies
    :class:`_JaxComputeRatesTransformer`, and compiles + exec-s the result
    into *module*'s namespace.  Returns the injected callable.
    """
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    compute_rates_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "computeRates":
            compute_rates_node = node
            break

    if compute_rates_node is None:
        raise UnsupportedFeatureError("OpenCOR module does not define computeRates")

    new_fn = copy.deepcopy(compute_rates_node)
    transformed = _JaxComputeRatesTransformer().visit(new_fn)
    assert isinstance(transformed, ast.FunctionDef)

    mod = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, str(source_path), "exec")
    exec(code, module.__dict__)
    return module.__dict__["_computeRates_jax"]


def compile_opencor_python_model(
    path: Path,
    backend: str,
    options: CompileOptions,
) -> CompiledModel:
    """Compile an OpenCOR-exported Python model into a :class:`~cellml2py.contracts.CompiledModel`.

    Parameters
    ----------
    path:
        Absolute path to the OpenCOR-exported ``.py`` file.
    backend:
        Numerical backend identifier.  ``"numpy"`` (default) or ``"jax"``.
        The JAX backend requires ``pip install cellml2py[jax]`` and returns a
        ``make_rhs()`` callable safe for ``jax.jit``, ``jax.grad``, and
        ``diffrax.ODETerm``.
    options:
        Compilation options including override target declarations.

    Returns
    -------
    CompiledModel
        Compiled model with RHS closure, layout metadata, initial state,
        and default parameter values mirroring the OpenCOR ``initConsts`` output.

    Raises
    ------
    UnsupportedFeatureError
        If the module is missing required functions or *backend* is unknown.
    MissingInputError
        If any override target cannot be resolved against the legend arrays.
    ImportError
        If *backend* is ``"jax"`` and JAX is not installed.
    """
    if backend not in ("numpy", "jax"):
        raise UnsupportedFeatureError(
            f"Unknown backend {backend!r}.  Choose 'numpy' or 'jax'."
        )

    module = _load_module_from_path(path)
    if not hasattr(module, "initConsts") or not hasattr(module, "computeRates"):
        raise UnsupportedFeatureError("OpenCOR module must provide initConsts and computeRates")

    init_states, init_constants = module.initConsts()
    init_states_arr = np.asarray(init_states, dtype=float)
    init_constants_arr = np.asarray(init_constants, dtype=float)

    legend_states, legend_algebraic, _legend_voi, legend_constants = module.createLegends()

    state_names, _state_q_to_idx, _state_short_to_idx = _build_names(legend_states, "state")
    algebraic_names, algebraic_q_to_idx, algebraic_short_to_idx = _build_names(
        legend_algebraic, "algebraic"
    )
    parameter_names, constant_q_to_idx, constant_short_to_idx = _build_names(
        legend_constants, "constant"
    )

    n_rates = init_states_arr.shape[0]
    resolved_overrides = [
        _resolve_override_target(
            spec,
            algebraic_q_to_idx,
            algebraic_short_to_idx,
            constant_q_to_idx,
            constant_short_to_idx,
            n_rates,
        )
        for spec in options.override_targets
    ]

    forcing_names = tuple(spec.target for spec in options.override_targets)

    # Expose qualified parameter names and short aliases when unambiguous.
    default_params: dict[str, float] = {}
    for i, q in enumerate(parameter_names):
        default_params[q] = float(init_constants_arr[i])
    for short_name, idx in constant_short_to_idx.items():
        default_params[short_name] = float(init_constants_arr[idx])

    parameter_key_to_idx: dict[str, int] = {}
    parameter_key_to_idx.update(constant_q_to_idx)
    parameter_key_to_idx.update(constant_short_to_idx)

    layout = RuntimeLayout(
        state_names=tuple(state_names),
        parameter_names=tuple(parameter_names),
        forcing_names=forcing_names,
        override_targets=options.override_targets,
    )

    # ------------------------------------------------------------------
    # JAX backend path
    # ------------------------------------------------------------------
    if backend == "jax":
        from .cellml_compiler import _try_import_jax  # lazy; raises if missing

        _try_import_jax()
        import jax.numpy as jnp  # type: ignore

        # Inject jnp into module so that _computeRates_jax (which calls
        # jnp.zeros internally after AST rewriting) can resolve it.
        module.__dict__["jnp"] = jnp
        jax_compute_rates = _build_jax_compute_rates(module, path)
        # Replace OpenCOR's math.pow-based power helper so it is JAX-traceable.
        module.__dict__["power"] = jnp.power

        _sanitize_nan: bool = getattr(options, "sanitize_nan", True)

        def _jax_rhs_builder():
            base_constants_jax = jnp.array(init_constants_arr, dtype=float)

            def rhs(t, x, args):
                """JAX-traceable RHS for this OpenCOR model."""
                params, forcing_values = CompiledModel._unpack_args(args)
                x_arr = jnp.asarray(x)

                if x_arr.shape != init_states_arr.shape:
                    raise ShapeError(
                        f"x must have shape {init_states_arr.shape}, got {x_arr.shape}"
                    )

                constants = base_constants_jax
                _params = params or {}
                for key, value in _params.items():
                    idx = parameter_key_to_idx.get(key)
                    if idx is None:
                        raise MissingInputError(f"Unknown parameter key: {key}")
                    constants = constants.at[idx].set(value)

                forcing_arr = jnp.asarray(
                    forcing_values if forcing_values is not None else (), dtype=float
                ).reshape(-1)
                if forcing_arr.shape[0] != len(resolved_overrides):
                    raise ShapeError(
                        "forcing vector length does not match override declarations: "
                        f"expected {len(resolved_overrides)}, got {forcing_arr.shape[0]}"
                    )

                # Constant overrides injected before computeRates.
                for forcing_i, ov in enumerate(resolved_overrides):
                    if ov.kind == "constant":
                        constants = constants.at[ov.index].set(forcing_arr[forcing_i])

                rates = jax_compute_rates(t, x_arr, constants)

                # Rate overrides injected after computeRates.
                for forcing_i, ov in enumerate(resolved_overrides):
                    if ov.kind == "rate":
                        rates = rates.at[ov.index].set(forcing_arr[forcing_i])

                if _sanitize_nan:
                    rates = jnp.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
                return rates

            return rhs

        return CompiledModel(
            backend="jax",
            layout=layout,
            initial_state=init_states_arr,
            default_params=default_params,
            _rhs_builder=_jax_rhs_builder,
        )

    # ------------------------------------------------------------------
    # NumPy backend path (default)
    # ------------------------------------------------------------------
    overridden_compute_rates = _build_overridden_compute_rates(module, path)

    def _rhs_builder():
        base_constants = init_constants_arr.copy()
        algebraic_override_map = {
            ov.index: forcing_i
            for forcing_i, ov in enumerate(resolved_overrides)
            if ov.kind == "algebraic"
        }

        def rhs(t: float, x: np.ndarray, args: object) -> np.ndarray:
            params, forcing_values = CompiledModel._unpack_args(args)
            x_arr = np.asarray(x, dtype=float)

            if x_arr.shape != init_states_arr.shape:
                raise ShapeError(
                    f"x must have shape {init_states_arr.shape}, got {x_arr.shape}"
                )

            constants = base_constants.copy()
            params = params or {}
            for key, value in params.items():
                idx = parameter_key_to_idx.get(key)
                if idx is None:
                    raise MissingInputError(f"Unknown parameter key: {key}")
                constants[idx] = float(value)

            forcing_arr = np.asarray(forcing_values if forcing_values is not None else (), dtype=float)
            forcing_arr = forcing_arr.reshape(-1)
            if forcing_arr.shape[0] != len(resolved_overrides):
                raise ShapeError(
                    "forcing vector length does not match override declarations: "
                    f"expected {len(resolved_overrides)}, got {forcing_arr.shape[0]}"
                )

            # Constant overrides are injected before computeRates executes.
            for forcing_i, ov in enumerate(resolved_overrides):
                if ov.kind == "constant":
                    constants[ov.index] = float(forcing_arr[forcing_i])

            if algebraic_override_map:
                rates = overridden_compute_rates(
                    float(t),
                    x_arr.tolist(),
                    constants.tolist(),
                    forcing_arr.tolist(),
                    algebraic_override_map,
                )
            else:
                rates = module.computeRates(float(t), x_arr.tolist(), constants.tolist())

            rates_arr = np.asarray(rates, dtype=float)

            # Rate overrides are injected after computeRates executes.
            for forcing_i, ov in enumerate(resolved_overrides):
                if ov.kind == "rate":
                    rates_arr[ov.index] = float(forcing_arr[forcing_i])

            return rates_arr

        return rhs

    return CompiledModel(
        backend=backend,
        layout=layout,
        initial_state=init_states_arr,
        default_params=default_params,
        _rhs_builder=_rhs_builder,
    )

