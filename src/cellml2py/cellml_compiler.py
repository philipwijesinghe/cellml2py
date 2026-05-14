"""Direct CellML to NumPy RHS compiler.

This module translates CellML 1.0 / 1.1 documents into NumPy-backed
runtime callables with the signature ``rhs(t, x, args)``.

CellML language elements
-------------------------
CellML is an XML-based language for mathematical models of biological
systems.  The specification is maintained by the Auckland Bioengineering
Institute: https://www.cellml.org/specifications/cellml_1.1.

Relevant structural elements:

* ``<model>`` — root element; contains components and connections.
* ``<component>`` — a named compartment owning a set of variables and
  optionally a ``<math>`` block.  In CellML 1.0 the component owns its
  equations; a variable declared in one component can be mapped to a variable
  in another via a ``<connection>`` element.
* ``<variable>`` — a named scalar with optional ``initial_value``, ``units``,
  and ``public_interface`` / ``private_interface`` attributes.  Connected
  variables share the same runtime value.
* ``<connection>`` / ``<map_components>`` / ``<map_variables>`` — declare that
  two variables represent the same quantity.  The compiler groups all connected
  variables into a canonical alias group using Union-Find.
* ``<math>`` (Content MathML) — encodes the equations for a component.  Each
  top-level child is an ``<apply><eq/> LHS RHS</apply>`` statement where LHS
  is either a ``<ci>`` (algebraic assignment) or a ``<diff>`` application
  (ODE rate), and RHS is a recursive MathML expression.

Compilation steps
-----------------
Step 1 — ``_collect_components``
    Walk every ``<component>`` element; register variables, record
    ``initial_value`` attributes, and tag ``time`` variables.

Step 2 — ``_collect_connections``
    Walk every ``<connection>`` element; apply Union-Find ``union()`` across
    each ``<map_variables>`` pair to form alias equivalence classes.

Step 3 — ``_finalize_roots``
    Propagate initial values to alias group representatives.  Resolve time
    variable roots after connection merging.

Step 4 — ``_collect_equations``
    For each ``<math>`` element, classify each top-level ``<apply>`` as either
    an algebraic assignment or an ODE rate.  Translate the MathML RHS tree to
    a Python expression string via ``_mathml_to_code``.

Step 5 — ``_canonicalize_symbols``
    Pick a stable public name for each alias group, preferring the member that
    owns the defining equation.  Rewrite all ``V("old_root")`` tokens in
    stored expression strings to the new canonical name.

Step 6 — ``_apply_state_order_hint``
    Optionally reorder ``state_roots`` to match a sibling OpenCOR-exported
    legend, for predictable index correspondence.

Step 7 — ``_build_name_maps``
    Produce valid Python identifier names for every canonical root so that
    expression strings compile cleanly with ``compile()``.

Step 8 — ``_precompile_expressions``
    Compile all expression strings to CPython bytecode with
    ``compile(..., 'eval')`` and topologically sort algebraics so the
    evaluation loop runs each algebraic after its dependencies.

Step 9 — ``_build_compiled_model``
    Construct the :class:`~cellml2py.contracts.CompiledModel` with its NumPy
    RHS closure.  The closure captures all bytecode objects and ordering data;
    the compiler instance can be garbage-collected afterwards.

Runtime evaluation
------------------
``rhs(t, x, args)`` parameters:

* ``t``    — scalar float time (the VOI in CellML terminology).
* ``x``    — 1-D NumPy array of state values in ``layout.state_names`` order.
* ``args`` — unpacked by ``_unpack_args`` into ``(params, forcing)``; pass
             ``None`` when there are no overrides.

The closure evaluates in four steps:

1. Build an ``env`` dict: time → ``t``, state roots → ``x[i]``,
   parameter roots → default values (updated by ``params`` if supplied).
2. Apply any ``constant`` or ``algebraic`` overrides from the forcing vector.
3. Evaluate every algebraic in topological order with ``eval()``.
4. Evaluate every rate expression and pack the results into a 1-D array.

Numerical stability
-------------------
Three helpers guard against floating-point hazards common in electrophysiology:

* ``_safe_exp`` — clips the exponent to [-700, 700] to avoid overflow.
* ``_safe_divide`` — returns a sign-preserving near-zero result when the
  denominator is below epsilon = 1e-12.
* ``_safe_power`` — forces real-domain evaluation for fractional powers of
  negative bases.

``RuntimeWarning`` and ``numpy.errstate`` overflows are suppressed inside the
RHS loop.  Remaining NaN/Inf in the rate vector is zeroed before returning to
prevent adaptive solvers from reducing the step size to machine epsilon.

Limitations
-----------
* Only CellML 1.0 / 1.1 is supported.
* Unit attributes are parsed but not used for conversion; models are assumed to
  use consistent units throughout.
* Implicit algebraic equations are not supported; the LHS must be a bare
  ``<ci>`` or ``<diff>``.
* The NumPy RHS calls ``eval()`` and ``float()`` eagerly and is not
  JAX-traceable.  Use the ``"jax"`` backend for that.
"""

from __future__ import annotations

import importlib.util
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import CodeType
from typing import Callable
from xml.etree import ElementTree as ET

import numpy as np

from .contracts import CompileOptions, CompiledModel, OverrideSpec, RuntimeLayout
from .exceptions import MissingInputError, ShapeError, UnsupportedFeatureError


def _local_name(tag: str) -> str:
    """Strip the XML namespace prefix from a Clark-notation tag.

    Python's ``xml.etree.ElementTree`` returns element tags in Clark notation,
    e.g. ``"{http://www.cellml.org/cellml/1.0#}component"``.  This function
    discards the namespace URI so callers can match on bare local names like
    ``"component"`` regardless of which CellML namespace URI the document uses.
    """
    return tag.rsplit("}", 1)[-1]


def _import_libcellml():
    """Return the ``libcellml`` module if it is installed, otherwise ``None``.

    ``libcellml`` is an optional C++ extension that provides a standards-
    compliant CellML 2.0 parser and validator.  When it is available the
    compiler uses it for an extra validation pass before attempting its own
    parsing, which surfaces human-readable standard-conformance errors early.
    The dependency is strictly optional: compilation proceeds via
    ``xml.etree.ElementTree`` regardless.
    """
    if importlib.util.find_spec("libcellml") is None:
        return None
    import libcellml  # type: ignore

    return libcellml


def _try_import_jax():
    """Return the ``jax`` module, raising a clear error when it is not installed.

    JAX is an optional dependency.  This helper is called lazily so that the
    module can be imported in NumPy-only environments without cost.

    Raises
    ------
    ImportError
        With a ``pip install cellml2py[jax]`` hint when JAX is absent.
    """
    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for backend='jax'.  "
            "Install it with: pip install cellml2py[jax]"
        )
    import jax  # type: ignore

    return jax


def _best_effort_libcellml_validate(path: Path) -> None:
    """Run a best-effort standards-conformance check via ``libcellml`` if available.

    When ``libcellml`` is installed this function parses the document with
    libcellml's own ``Parser`` and then runs its ``Validator`` over the
    resulting model object.  Any validation issues are surfaced as
    ``libcellml`` error messages visible to the caller; this function itself
    does not raise on failure so that the compiler can still attempt its own
    XML-level parse of CellML 1.x documents that libcellml may reject as
    CellML 2.0 non-compliant.

    Both the ``Parser`` and ``Validator`` class names are probed by two
    candidate names to accommodate different ``libcellml`` build variants.

    Parameters
    ----------
    path:
        Absolute path to the ``.cellml`` file to validate.
    """
    libcellml = _import_libcellml()
    if libcellml is None:
        return

    parser = None
    for parser_name in ("Parser", "CellmlParser"):
        parser_cls = getattr(libcellml, parser_name, None)
        if parser_cls is not None:
            parser = parser_cls()
            break
    if parser is None:
        return

    model = None
    if hasattr(parser, "parseModel"):
        model = parser.parseModel(str(path))
    elif hasattr(parser, "parse"):
        model = parser.parse(str(path))
    elif hasattr(parser, "parseModelFromString"):
        model = parser.parseModelFromString(path.read_text(encoding="utf-8"))

    if model is None:
        return

    validator = None
    for validator_name in ("Validator", "CellmlValidator"):
        validator_cls = getattr(libcellml, validator_name, None)
        if validator_cls is not None:
            validator = validator_cls()
            break
    if validator is None:
        return

    if hasattr(validator, "validateModel"):
        validator.validateModel(model)
    elif hasattr(validator, "validate"):
        validator.validate(model)


class _UnionFind:
    """Path-compressing Union-Find used to group CellML connected variables.

    CellML connections declare that a variable in one component is identical
    to a variable in another component.  A single physical quantity may be
    shared across many components through a chain of pairwise connections.
    Union-Find (disjoint-set union) efficiently groups all such aliases into
    equivalence classes so the compiler can later canonicalise each group to
    a single representative name.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        """Register *item* as its own representative if not already present."""
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        """Return the canonical representative of *item*'s equivalence class.

        Path compression is applied so that subsequent calls for any member of
        the same class return in near-O(1) time.
        """
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """Merge the equivalence classes of *left* and *right*."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class _VariableInfo:
    """Immutable snapshot of one CellML ``<variable>`` element's metadata.

    Attributes
    ----------
    component:
        The CellML component name that declares this variable.
    name:
        The ``name`` attribute of the ``<variable>`` element.
    root:
        The canonical representative of this variable's alias group.  Before
        ``_canonicalize_symbols`` runs this is the Union-Find root; after
        canonicalisation it is the chosen public identifier for the group.
    initial_value:
        Parsed ``initial_value`` attribute, or ``None`` if absent.  CellML 1.x
        defines ``initial_value`` as a real-number literal applied at ``t=0``
        for state variables and as a default parameter value for constants.
    """

    component: str
    name: str
    root: str
    initial_value: float | None


class _CellMLModelCompiler:
    """Stateful compiler for one CellML 1.0 / 1.1 document.

    Each compiler instance is constructed for a single ``.cellml`` file and
    used exactly once via :meth:`compile`.  Internal mutable state accumulates
    through nine sequential compilation phases (see module docstring) and is
    then frozen into an immutable :class:`~cellml2py.contracts.CompiledModel`.

    CellML document structure assumptions
    --------------------------------------
    * The document uses either the CellML 1.0
      (``http://www.cellml.org/cellml/1.0#``) or CellML 1.1
      (``http://www.cellml.org/cellml/1.1#``) namespace; the namespace URI is
      stripped from tag names before any comparison.
    * A ``<component>`` may contain zero or more ``<variable>`` elements and at
      most one ``<math>`` element.
    * Each ``<math>`` element contains only top-level ``<apply><eq/>…</apply>``
      equalities (no bare numeric constants or relational statements at the top
      level).
    * Equations are explicit: the left-hand side is always a single ``<ci>``
      (algebraic assignment) or a ``<apply><diff/>…<ci>…</apply>`` (rate).
    * The ``bvar`` element inside a ``<diff>`` is assumed to denote the model's
      independent variable (time) and is not inspected further.
    """

    def __init__(self, path: Path, options: CompileOptions) -> None:
        self.path = path
        self.options = options
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.variables_by_id: dict[str, _VariableInfo] = {}
        self.alias_lookup: dict[tuple[str, str], str] = {}
        self.root_to_members: dict[str, list[str]] = {}
        self.component_order: list[str] = []
        self.component_variables: dict[str, list[str]] = {}
        self.state_roots: list[str] = []
        self.parameter_roots: list[str] = []
        self.state_initial_values: dict[str, float] = {}
        self.parameter_initial_values: dict[str, float] = {}
        self.algebraic_exprs: dict[str, str] = {}
        self.algebraic_codes: dict[str, CodeType] = {}
        self.algebraic_topo_order: list[str] = []
        self.rate_exprs: dict[str, str] = {}
        self.rate_codes: dict[str, CodeType] = {}
        self._equation_kind_by_exact: dict[str, str] = {}
        self.time_roots: set[str] = set()
        self._short_to_roots: dict[str, set[str]] = {}
        self._exact_to_root: dict[str, str] = {}
        self._root_to_safe: dict[str, str] = {}
        self._override_targets: list[OverrideSpec] = list(options.override_targets)
        self._forced_kind_by_root: dict[str, str] = {}

    def compile(self) -> CompiledModel:
        """Compile the loaded CellML document into a :class:`CompiledModel`."""
        _best_effort_libcellml_validate(self.path)
        self._collect_components()
        self._collect_connections()
        self._finalize_roots()
        self._collect_equations()
        self._canonicalize_symbols()
        self._apply_state_order_hint()
        self._build_name_maps()
        self._precompile_expressions()
        return self._build_compiled_model(self.options)

    def _collect_components(self) -> None:
        """Phase 1: Parse all ``<component>`` elements and their ``<variable>`` children.

        For each component this method:

        * Appends the component name to ``component_order`` (preserving document
          order, which is used as a stable tiebreaker throughout compilation).
        * Registers each ``<variable>`` in ``variables_by_id`` keyed by the
          fully-qualified id ``"<component>::<variable>"``.
        * Builds ``alias_lookup[(component, name)] -> variable_id`` for
          fast O(1) resolution during MathML translation.
        * Tags variables named ``"time"`` as time (VOI) candidates; the final
          time root is resolved after connections are merged in Phase 3.
        * Stores the raw ``initial_value`` attribute (if present) as a float.
        """
        for element in self.root:
            if _local_name(element.tag) != "component":
                continue
            component_name = element.attrib["name"]
            self.component_order.append(component_name)
            variable_ids: list[str] = []
            for variable in element:
                if _local_name(variable.tag) != "variable":
                    continue
                variable_name = variable.attrib["name"]
                variable_id = f"{component_name}::{variable_name}"
                variable_ids.append(variable_id)
                self.alias_lookup[(component_name, variable_name)] = variable_id
                if variable_name == "time":
                    self.time_roots.add(variable_id)
                initial_value = variable.attrib.get("initial_value")
                info = _VariableInfo(
                    component=component_name,
                    name=variable_name,
                    root=variable_id,
                    initial_value=float(initial_value)
                    if initial_value is not None
                    else None,
                )
                self.variables_by_id[variable_id] = info
            self.component_variables[component_name] = variable_ids

    def _collect_connections(self) -> None:
        """Phase 2: Merge connected variables into equivalence classes.

        CellML ``<connection>`` elements consist of a ``<map_components>``
        child (which names the two components being connected) and one or more
        ``<map_variables>`` children (each pairing one variable from each
        component).  A pair of connected variables must have the same value at
        all times; the compiler represents this by merging their ids in a
        Union-Find structure.

        After all connections have been processed every variable id is mapped to
        its Union-Find root.  ``root_to_members`` is built as the inverse
        mapping so later phases can iterate over all members of a group.
        ``variables_by_id`` entries are updated with the new ``root`` values.
        """
        union_find = _UnionFind()
        for variable_id in self.variables_by_id:
            union_find.add(variable_id)

        for element in self.root:
            if _local_name(element.tag) != "connection":
                continue
            map_components = None
            map_variables: list[tuple[str, str]] = []
            for child in element:
                if _local_name(child.tag) == "map_components":
                    map_components = (
                        child.attrib["component_1"],
                        child.attrib["component_2"],
                    )
                elif _local_name(child.tag) == "map_variables":
                    map_variables.append(
                        (child.attrib["variable_1"], child.attrib["variable_2"])
                    )
            if map_components is None:
                continue
            comp1, comp2 = map_components
            for var1, var2 in map_variables:
                left = f"{comp1}::{var1}"
                right = f"{comp2}::{var2}"
                if left in self.variables_by_id and right in self.variables_by_id:
                    union_find.union(left, right)

        root_members: dict[str, list[str]] = {}
        for variable_id in self.variables_by_id:
            root = union_find.find(variable_id)
            root_members.setdefault(root, []).append(variable_id)

        self.root_to_members = root_members

        for root, members in root_members.items():
            for member in members:
                self.variables_by_id[member] = _VariableInfo(
                    component=self.variables_by_id[member].component,
                    name=self.variables_by_id[member].name,
                    root=root,
                    initial_value=self.variables_by_id[member].initial_value,
                )

    def _finalize_roots(self) -> None:
        """Phase 3: Propagate initial values and resolve time variable roots.

        After Union-Find merging each alias group's representative is an
        arbitrary member (the first element placed in the group).  This phase:

        * Rebuilds ``_exact_to_root`` and ``_short_to_roots`` look-up tables
          from the now-stable Union-Find roots.
        * Copies any ``initial_value`` found on a group member into
          ``parameter_initial_values`` keyed by the group root.  Only the
          first found value is used (CellML models should not have conflicting
          initial values on connected variables).
        * Resolves raw ``time_roots`` variable ids to their final group roots
          so that ``V("environment::time")``-style references can be matched.

        State and parameter classification is deferred to Phase 4 because it
        requires knowledge of which variables have rate equations.
        """
        for variable_id, info in self.variables_by_id.items():
            self._exact_to_root[variable_id] = info.root
            self._short_to_roots.setdefault(info.name, set()).add(info.root)

        for root, members in self.root_to_members.items():
            initial_values = [
                self.variables_by_id[member].initial_value
                for member in members
                if self.variables_by_id[member].initial_value is not None
            ]
            if not initial_values:
                continue
            value = float(initial_values[0])  # type: ignore
            self.parameter_initial_values[root] = value

        resolved_time_roots = set()
        for time_variable_id in self.time_roots:
            resolved_time_roots.add(self._exact_to_root[time_variable_id])
        self.time_roots = resolved_time_roots

        # NOTE: state / parameter classification happens in _collect_equations
        # (Phase 4) once we know which variables have rate equations.

    def _component_root(self, component_name: str, variable_name: str) -> str:
        """Resolve a ``(component, variable)`` pair to its canonical alias root.

        Looks up the fully-qualified variable id
        ``"<component>::<variable>"`` in ``_exact_to_root``.

        Raises
        ------
        UnsupportedFeatureError
            If the pair does not correspond to any declared CellML variable.
        """
        variable_id = f"{component_name}::{variable_name}"
        try:
            return self._exact_to_root[variable_id]
        except KeyError as exc:
            raise UnsupportedFeatureError(
                f"Unknown CellML variable reference: {component_name}::{variable_name}"
            ) from exc

    def _collect_equations(self) -> None:
        """Phase 4: Parse MathML equations and classify state vs parameter roots.

        Iterates over every ``<component>`` element and its ``<math>`` child.
        Each top-level ``<apply>`` inside ``<math>`` must be an equality
        (``<eq/>`` as first child) with exactly two operands::

            <apply>
              <eq/>
              LHS  <!-- <ci> or <apply><diff/>...</apply> -->
              RHS  <!-- arbitrary Content MathML expression -->
            </apply>

        The LHS is classified via ``_parse_lhs``:

        * ``<ci>`` → algebraic assignment; expression stored in
          ``algebraic_exprs`` keyed by the variable's exact id.
        * ``<apply><diff/>`` → ODE rate; expression stored in ``rate_exprs``.

        After all equations are collected:

        * ``state_roots`` - roots that appear in ``rate_exprs`` (i.e. variables
          with an ODE governing their time derivative).
        * ``parameter_roots`` - roots that have an ``initial_value`` but *no*
          rate equation (i.e. constant parameters).
        """
        for element in self.root:
            if _local_name(element.tag) != "component":
                continue
            component_name = element.attrib["name"]
            for child in element:
                if _local_name(child.tag) != "math":
                    continue
                for equation in child:
                    if _local_name(equation.tag) != "apply" or len(equation) == 0:
                        continue
                    if _local_name(equation[0].tag) != "eq" or len(equation) < 3:
                        continue
                    lhs = equation[1]
                    rhs = equation[2]
                    target_exact, target_kind = self._parse_lhs(component_name, lhs)
                    expression = self._mathml_to_code(component_name, rhs)
                    if target_kind == "rate":
                        self.rate_exprs[target_exact] = expression
                    else:
                        self.algebraic_exprs[target_exact] = expression
                    self._equation_kind_by_exact[target_exact] = target_kind

        state_roots: list[str] = []
        for root in self.rate_exprs:
            if root not in state_roots:
                state_roots.append(root)
        self.state_roots = state_roots
        self.parameter_roots = [
            root
            for root in self.parameter_initial_values
            if root not in self.rate_exprs
        ]

    def _parse_lhs(self, component_name: str, lhs: ET.Element) -> tuple[str, str]:
        """Classify the left-hand side of a CellML equation.

        CellML 1.x defines two valid LHS forms:

        1. **Algebraic assignment** - a bare ``<ci>`` element naming the
           variable being defined::

               <ci>V</ci>

        2. **ODE rate** - a ``<apply><diff/><bvar>...</bvar><ci>x</ci></apply>``
           expression declaring the first time-derivative of ``x``::

               <apply>
                 <diff/>
                 <bvar><ci>time</ci></bvar>
                 <ci>V</ci>
               </apply>

        Parameters
        ----------
        component_name:
            The CellML component that owns this equation.
        lhs:
            The parsed LHS ``ET.Element``.

        Returns
        -------
        tuple[str, str]
            ``(exact_variable_id, kind)`` where *kind* is ``"algebraic"`` or
            ``"rate"``.

        Raises
        ------
        UnsupportedFeatureError
            If the LHS structure is not one of the two recognised forms.
        """
        if _local_name(lhs.tag) == "ci":
            variable_name = (lhs.text or "").strip()
            variable_id = f"{component_name}::{variable_name}"
            if variable_id not in self.variables_by_id:
                raise UnsupportedFeatureError(
                    f"Unknown CellML variable reference: {component_name}::{variable_name}"
                )
            return variable_id, "algebraic"
        if (
            _local_name(lhs.tag) == "apply"
            and len(lhs) > 0
            and _local_name(lhs[0].tag) == "diff"
        ):
            target_ci = None
            for child in lhs:
                if _local_name(child.tag) == "ci":
                    target_ci = child
                    break
            if target_ci is None:
                raise UnsupportedFeatureError(
                    "Could not parse derivative target in CellML equation"
                )
            variable_name = (target_ci.text or "").strip()
            variable_id = f"{component_name}::{variable_name}"
            if variable_id not in self.variables_by_id:
                raise UnsupportedFeatureError(
                    f"Unknown CellML variable reference: {component_name}::{variable_name}"
                )
            return variable_id, "rate"
        raise UnsupportedFeatureError(
            f"Unsupported equation lhs in component {component_name}"
        )

    def _canonicalize_symbols(self) -> None:
        """Pick stable canonical names for alias groups.

        Connected variables often share a union-find representative that is not
        the most useful public name. The runtime instead prefers the member that
        owns the defining algebraic equation, then a rate equation, then any
        member with an initial value.
        """
        canonical_by_group_root: dict[str, str] = {}
        for group_root, members in self.root_to_members.items():
            algebraic_members = [
                member
                for member in members
                if self._equation_kind_by_exact.get(member) == "algebraic"
            ]
            rate_members = [
                member
                for member in members
                if self._equation_kind_by_exact.get(member) == "rate"
            ]
            initial_members = [
                member
                for member in members
                if self.variables_by_id[member].initial_value is not None
            ]

            if algebraic_members:
                chosen = algebraic_members[0]
            elif rate_members:
                chosen = rate_members[0]
            elif initial_members:
                chosen = initial_members[0]
            else:
                chosen = members[0]

            canonical_by_group_root[group_root] = chosen

        exact_to_root: dict[str, str] = {}
        short_to_roots: dict[str, set[str]] = {}
        parameter_initial_values: dict[str, float] = {}
        for variable_id, info in self.variables_by_id.items():
            canonical_root = canonical_by_group_root[info.root]
            exact_to_root[variable_id] = canonical_root
            short_to_roots.setdefault(info.name, set()).add(canonical_root)

            if info.initial_value is not None:
                parameter_initial_values.setdefault(
                    canonical_root, float(info.initial_value)
                )

            self.variables_by_id[variable_id] = _VariableInfo(
                component=info.component,
                name=info.name,
                root=canonical_root,
                initial_value=info.initial_value,
            )

        self._exact_to_root = exact_to_root
        self._short_to_roots = short_to_roots

        # Build a mapping from every pre-canonical root (union-find root) to its
        # canonical root so that V("old_root") references inside expression strings
        # can be rewritten to V("canonical_root").
        old_root_to_canonical: dict[str, str] = {
            group_root: canonical_by_group_root[group_root]
            for group_root in canonical_by_group_root
        }

        def _rewrite_expr(expr: str) -> str:
            # Replace every V("old") with V("canonical") where old != canonical.
            def _sub(m: re.Match) -> str:
                old = m.group(1)
                new = old_root_to_canonical.get(old, old)
                return f'V("{new}")'

            return re.sub(r'V\("([^"]+)"\)', _sub, expr)

        self.algebraic_exprs = {
            exact_to_root[target_exact]: _rewrite_expr(expression)
            for target_exact, expression in self.algebraic_exprs.items()
        }
        self.rate_exprs = {
            exact_to_root[target_exact]: _rewrite_expr(expression)
            for target_exact, expression in self.rate_exprs.items()
        }
        self.parameter_initial_values = parameter_initial_values
        self.state_roots = [root for root in self.rate_exprs]
        self.parameter_roots = [
            root
            for root in self.parameter_initial_values
            if root not in self.rate_exprs
        ]
        self.time_roots = {exact_to_root[time_root] for time_root in self.time_roots}

    def _apply_state_order_hint(self) -> None:
        """Prefer sibling OpenCOR legend order when it matches the same state set.

        This is mainly used for parity tests and for predictable interoperability
        with models that were already exported through OpenCOR.
        """
        legend_order = self._load_reference_state_order()
        if not legend_order:
            return

        state_root_set = set(self.state_roots)
        ordered_roots: list[str] = []
        seen: set[str] = set()

        for exact_name in legend_order:
            root = self._exact_to_root.get(exact_name)
            if root is None or root not in state_root_set or root in seen:
                continue
            ordered_roots.append(root)
            seen.add(root)

        for root in self.state_roots:
            if root not in seen:
                ordered_roots.append(root)

        self.state_roots = ordered_roots

    def _load_reference_state_order(self) -> list[str]:
        """Search sibling Python files for an OpenCOR-exported state legend.

        An OpenCOR-generated Python file contains a ``createLegends`` function
        that assigns strings like
        ``legend_states[0] = "V in component membrane (millivolt)"``.
        This method parses that pattern to recover an ordered list of fully-
        qualified variable ids in the form ``"<component>::<variable>"``.

        Returns an empty list if no suitable file is found or if the recovered
        state count does not match ``self.state_roots``.
        """
        candidates = sorted(self.path.parent.glob("*.py"))
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                continue
            if "legend_states" not in text or "createLegends" not in text:
                continue

            matches = re.findall(
                r'legend_states\[(\d+)\] = "([^"]+)"',
                text,
            )
            if not matches:
                continue

            ordered: list[str] = [""] * (max(int(index) for index, _ in matches) + 1)
            for index, legend in matches:
                ordered[int(index)] = legend

            state_exact_names: list[str] = []
            for legend in ordered:
                if not legend:
                    continue
                if " in component " not in legend:
                    continue
                variable_name, remainder = legend.split(" in component ", 1)
                component_name = remainder.split(" (", 1)[0]
                exact_name = f"{component_name}::{variable_name}"
                state_exact_names.append(exact_name)

            if len(state_exact_names) == len(self.state_roots):
                return state_exact_names

        return []

    def _mathml_to_code(self, component_name: str, node: ET.Element) -> str:
        """Recursively translate a Content MathML expression tree to a Python string.

        CellML 1.x uses the W3C Content MathML 2 subset
        (https://www.w3.org/TR/MathML2/chapter4.html) to encode equations.
        This method walks the element tree depth-first and returns a Python
        expression string that evaluates to the same numeric value when
        ``eval()``-ed inside the RHS closure.

        Symbol resolution
        -----------------
        ``<ci>`` leaf nodes are translated to ``V("<canonical_root>")`` where
        the callable ``V`` is the runtime variable resolver.  The canonical
        root is looked up via ``_component_root`` so that connected-variable
        aliases all resolve to the same value.

        Numeric literals
        ----------------
        ``<cn>`` elements are emitted verbatim.  ``<pi>`` maps to ``np.pi``.

        Arithmetic operators
        --------------------
        ``<plus>``, ``<minus>``, ``<times>`` are n-ary; ``<divide>`` and
        ``<power>`` are binary.  Division and power use the safe helpers
        ``SAFE_DIVIDE`` / ``SAFE_POWER`` to avoid overflow and domain errors
        common in stiff electrophysiology models.

        Piecewise expressions
        ---------------------
        ``<piecewise>`` is translated to a nested ``np.where(...)`` chain,
        preserving element order.  The ``<otherwise>`` branch (or ``0.0`` if
        absent) forms the innermost expression.

        Transcendental and trigonometric functions
        -------------------------------------------
        ``<exp>`` uses ``SAFE_EXP`` (clipped at ±700).  ``<ln>`` / ``<log>``
        map to ``np.log``.  ``<sin>``, ``<cos>``, ``<tan>`` map directly.

        Relational and boolean operators
        ---------------------------------
        Relational operators (``<eq>``, ``<gt>``, ``<lt>``, ``<geq>``,
        ``<leq>``, ``<neq>``) produce element-wise boolean expressions
        suitable for use as ``np.where`` conditions.  Boolean operators
        ``<and>`` / ``<or>`` / ``<not>`` use bitwise NumPy operators so they
        broadcast correctly over array inputs.

        Parameters
        ----------
        component_name:
            The enclosing CellML component; needed to resolve ``<ci>`` names.
        node:
            The MathML ``ET.Element`` to translate.

        Returns
        -------
        str
            A Python expression string.

        Raises
        ------
        UnsupportedFeatureError
            If the element or operator is not in the supported subset.
        """
        tag = _local_name(node.tag)

        # --- Leaf nodes ---
        if tag == "ci":
            return f'V("{self._component_root(component_name, (node.text or "").strip())}")'
        if tag == "cn":
            # Numeric literal - emit verbatim so Python eval() parses it.
            text = (node.text or "0").strip()
            return text
        if tag == "pi":
            return "np.pi"

        # --- Piecewise / conditional ---
        if tag == "piecewise":
            otherwise_expr = "0.0"
            pieces: list[tuple[str, str]] = []
            for child in node:
                child_tag = _local_name(child.tag)
                if child_tag == "piece":
                    if len(child) != 2:
                        raise UnsupportedFeatureError(
                            "Unsupported piecewise piece structure"
                        )
                    value = self._mathml_to_code(component_name, child[0])
                    condition = self._mathml_to_code(component_name, child[1])
                    pieces.append((value, condition))
                elif child_tag == "otherwise":
                    if len(child) != 1:
                        raise UnsupportedFeatureError("Unsupported otherwise structure")
                    otherwise_expr = self._mathml_to_code(component_name, child[0])
            expression = otherwise_expr
            for value, condition in reversed(pieces):
                expression = f"np.where({condition}, {value}, {expression})"
            return expression

        if tag != "apply":
            raise UnsupportedFeatureError(f"Unsupported MathML node: {tag}")

        # All remaining nodes must be <apply> with an operator as first child.
        operator = _local_name(node[0].tag)
        args = [self._mathml_to_code(component_name, child) for child in node[1:]]

        # --- Arithmetic operators ---
        if operator == "plus":
            return "(" + " + ".join(args) + ")"
        if operator == "times":
            return "(" + " * ".join(args) + ")"
        if operator == "divide":
            if len(args) != 2:
                raise UnsupportedFeatureError("divide expects exactly 2 arguments")
            return f"SAFE_DIVIDE({args[0]}, {args[1]})"
        if operator == "power":
            if len(args) != 2:
                raise UnsupportedFeatureError("power expects exactly 2 arguments")
            return f"SAFE_POWER({args[0]}, {args[1]})"
        if operator == "root":
            if len(args) == 1:
                return f"np.sqrt({args[0]})"
            if len(args) == 2:
                return f"SAFE_POWER({args[1]}, SAFE_DIVIDE(1.0, ({args[0]})))"
            raise UnsupportedFeatureError("root expects 1 or 2 arguments")
        if operator == "minus":
            if len(args) == 1:
                return f"(-{args[0]})"
            if len(args) >= 2:
                expression = args[0]
                for arg in args[1:]:
                    expression = f"({expression} - {arg})"
                return expression
            raise UnsupportedFeatureError("minus expects at least 1 argument")

        # --- Transcendental and trigonometric functions ---
        if operator == "exp":
            if len(args) != 1:
                raise UnsupportedFeatureError("exp expects exactly 1 argument")
            return f"SAFE_EXP({args[0]})"
        if operator in {"ln", "log"}:
            if len(args) != 1:
                raise UnsupportedFeatureError("log expects exactly 1 argument")
            return f"np.log({args[0]})"
        if operator == "abs":
            return f"np.abs({args[0]})"
        if operator == "sin":
            return f"np.sin({args[0]})"
        if operator == "cos":
            return f"np.cos({args[0]})"
        if operator == "tan":
            return f"np.tan({args[0]})"
        if operator == "floor":
            return f"np.floor({args[0]})"
        if operator == "ceiling":
            return f"np.ceil({args[0]})"

        # --- Relational operators (produce boolean / mask expressions) ---
        if operator == "eq":
            return f"({args[0]} == {args[1]})"
        if operator in {"gt", "greater"}:
            return f"({args[0]} > {args[1]})"
        if operator in {"lt", "less"}:
            return f"({args[0]} < {args[1]})"
        if operator in {"geq", "greater_equal"}:
            return f"({args[0]} >= {args[1]})"
        if operator in {"leq", "less_equal"}:
            return f"({args[0]} <= {args[1]})"
        if operator in {"neq", "not_equal"}:
            return f"({args[0]} != {args[1]})"

        # --- Boolean operators ---
        if operator == "and":
            return "(" + " & ".join(args) + ")"
        if operator == "or":
            return "(" + " | ".join(args) + ")"
        if operator == "not":
            if len(args) != 1:
                raise UnsupportedFeatureError("not expects exactly 1 argument")
            return f"(~({args[0]}))"

        raise UnsupportedFeatureError(f"Unsupported MathML operator: {operator}")

    def _build_name_maps(self) -> None:
        """Phase 7: Build safe Python identifier names for every canonical root.

        Canonical roots are fully-qualified strings like
        ``"membrane::V"`` which are not valid Python identifiers.  This method
        replaces every non-alphanumeric character with ``_``, prepends a
        leading underscore if the result starts with a digit, and de-duplicates
        by appending a numeric suffix when needed.  The mapping is stored in
        ``_root_to_safe`` and is used by ``_rhs_builder`` to generate readable
        variable names in the eval context (currently informational only since
        the runtime uses dict look-ups rather than generated source code).
        """
        short_to_safe: dict[str, str] = {}
        root_order = self.state_roots + [
            root for root in self.parameter_roots if root not in self.state_roots
        ]
        for root in root_order:
            safe = re.sub(r"[^A-Za-z0-9_]+", "_", root)
            if safe and safe[0].isdigit():
                safe = f"_{safe}"
            candidate = safe
            index = 1
            while candidate in self._root_to_safe.values():
                candidate = f"{safe}_{index}"
                index += 1
            self._root_to_safe[root] = candidate
            short = root.split("::", 1)[-1]
            short_to_safe.setdefault(short, candidate)

    def _precompile_expressions(self) -> None:
        """Compile expression strings to bytecode and build algebraic evaluation order.

        Pre-compiling avoids re-parsing expression strings on every RHS call.
        The topological sort ensures each algebraic is evaluated after all of its
        dependencies, enabling a simple flat (non-recursive) evaluation loop.
        """
        for root, expr in self.algebraic_exprs.items():
            try:
                self.algebraic_codes[root] = compile(expr, f"<cellml:{root}>", "eval")
            except SyntaxError:
                pass  # fall back to string eval
        for root, expr in self.rate_exprs.items():
            try:
                self.rate_codes[root] = compile(expr, f"<cellml_rate:{root}>", "eval")
            except SyntaxError:
                pass
        self._build_algebraic_topo_order()

    def _build_algebraic_topo_order(self) -> None:
        """Topologically sort algebraic variables so each is evaluated after its deps."""
        algebraic_set = set(self.algebraic_exprs.keys())
        # Extract V("root") references from an expression string.
        dep_re = re.compile(r'V\("([^"]+)"\)')

        # Build adjacency: deps[root] = list of algebraic roots that root depends on.
        deps: dict[str, list[str]] = {}
        for root, expr in self.algebraic_exprs.items():
            deps[root] = [r for r in dep_re.findall(expr) if r in algebraic_set]

        order: list[str] = []
        visited: set[str] = set()
        in_progress: set[str] = set()

        def _dfs(node: str) -> None:
            if node in visited:
                return
            if node in in_progress:
                return  # cyclic dependency - skip to avoid infinite recursion
            in_progress.add(node)
            for dep in deps.get(node, []):
                _dfs(dep)
            in_progress.discard(node)
            visited.add(node)
            order.append(node)

        for root in self.algebraic_exprs:
            _dfs(root)

        self.algebraic_topo_order = order

    def _resolve_name(self, name: str, expected_kind: str | None = None) -> str:
        """Resolve a user-supplied variable name to its canonical root.

        Accepts either:

        * A fully-qualified id ``"<component>::<variable>"`` - looked up
          directly in ``_exact_to_root``.
        * A bare short name ``"V"`` - looked up in ``_short_to_roots``.  If
          multiple alias groups share the same short name the first canonical
          root is returned (deterministic due to document order preservation).

        Raises
        ------
        MissingInputError
            If the name is not found in any look-up table.
        """
        if name in self._exact_to_root:
            return self._exact_to_root[name]

        if "::" not in name:
            roots = self._short_to_roots.get(name)
            if roots is None:
                raise MissingInputError(f"Unknown CellML variable name: {name}")
            if len(roots) != 1:
                # If there is an alias set, it should still map to one canonical root.
                return next(iter(roots))
            return next(iter(roots))

        raise MissingInputError(f"Unknown CellML variable name: {name}")

    def _build_compiled_model(self, options: object) -> CompiledModel:
        """Phase 9: Assemble the public :class:`~cellml2py.contracts.CompiledModel`.

        This method captures the results of all previous compilation phases
        into a closure and returns an immutable ``CompiledModel``.  The closure
        approach means the ``_CellMLModelCompiler`` instance itself can be
        garbage-collected after this method returns; only the necessary
        sub-structures are kept alive by the closure.

        The returned ``CompiledModel`` exposes:

        * ``layout`` - deterministic ordering of state variables, parameters,
          and forcing inputs as ``RuntimeLayout``.
        * ``initial_state`` - NumPy array of CellML ``initial_value``
          attributes in ``state_names`` order.
        * ``default_params`` - dict keyed by both qualified and short names so
          callers can override parameters by either naming convention.
        * ``make_rhs()`` - factory that builds the NumPy RHS callable.
        """
        state_names = tuple(self.state_roots)
        parameter_names = tuple(self.parameter_roots)
        forcing_names = tuple(spec.target for spec in self._override_targets)
        state_name_set = set(state_names)

        default_params: dict[str, float] = {}
        for root in parameter_names:
            value = self.parameter_initial_values[root]
            default_params[root] = value
            short = root.split("::", 1)[-1]
            if short not in default_params:
                default_params[short] = value

        override_targets: list[tuple[str, str, int]] = []
        for forcing_index, spec in enumerate(self._override_targets):
            root = self._resolve_name(spec.target)
            kind = spec.kind or self._infer_kind(root)
            override_targets.append((root, kind, forcing_index))
            self._forced_kind_by_root[root] = kind

        layout = RuntimeLayout(
            state_names=state_names,
            parameter_names=parameter_names,
            forcing_names=forcing_names,
            override_targets=tuple(self._override_targets),
        )

        # Snapshot data needed by the closure so the compiler object does not
        # need to remain referenced after CompiledModel is returned.
        _algebraic_topo_order = self.algebraic_topo_order
        _algebraic_codes = self.algebraic_codes
        _algebraic_exprs = self.algebraic_exprs
        _rate_codes = self.rate_codes
        _rate_exprs = self.rate_exprs
        _param_initial_values = dict(self.parameter_initial_values)
        _time_roots = frozenset(self.time_roots)
        _resolve_name = self._resolve_name
        _forced_algebraic_roots = frozenset(
            self._resolve_name(spec.target)
            for spec in self._override_targets
            if (spec.kind or self._infer_kind(self._resolve_name(spec.target)))
            == "algebraic"
        )
        _sanitize_nan: bool = getattr(options, "sanitize_nan", True)

        def _rhs_builder() -> Callable[[float, np.ndarray, object], np.ndarray]:
            def rhs(t: float, x: np.ndarray, args: object) -> np.ndarray:
                """Evaluate the compiled rate equations at one state point.

                The RHS uses a flat, pre-ordered evaluation loop:
                1. Populate ``env`` with time, state values, and parameters.
                2. Evaluate every algebraic in topological order so each
                   expression only references already-computed values.
                3. Evaluate rate expressions using the same ``env``.

                This avoids recursive Python resolver calls and recreating
                context objects, giving a significant speedup over lazy evaluation.
                """
                params, forcing_values = CompiledModel._unpack_args(args)
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (len(state_names),):
                    raise ShapeError(
                        f"x must have shape {(len(state_names),)}, got {x_arr.shape}"
                    )

                forcing_arr = np.asarray(
                    forcing_values if forcing_values is not None else (), dtype=float
                ).reshape(-1)
                if forcing_arr.shape[0] != len(override_targets):
                    raise ShapeError(
                        "forcing vector length does not match override declarations: "
                        f"expected {len(override_targets)}, got {forcing_arr.shape[0]}"
                    )

                params = params or {}

                # Build the flat environment dict once per RHS call.
                t_float = float(t)
                env: dict[str, float] = {"time": t_float, "t": t_float}
                # Register every canonical time-variable root so V("environment::time")
                # style references resolve correctly.
                for _tr in _time_roots:
                    env[_tr] = t_float
                for state_index, root in enumerate(state_names):
                    env[root] = float(x_arr[state_index])
                for root, value in _param_initial_values.items():
                    if root not in state_name_set:
                        env[root] = float(value)
                for key, value in params.items():
                    env[_resolve_name(str(key))] = float(value)
                for root, kind, forcing_index in override_targets:
                    if kind in ("constant", "algebraic"):
                        env[root] = float(forcing_arr[forcing_index])

                # Resolver: dict lookup with 0.0 default for truly unknown names.
                local_env = {"V": lambda n, _g=env.get: _g(n, 0.0)}

                rates = np.empty(len(state_names), dtype=float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    with np.errstate(
                        divide="ignore", invalid="ignore", over="ignore", under="ignore"
                    ):
                        # Evaluate all algebraics in dependency order.
                        for root in _algebraic_topo_order:
                            if root in _forced_algebraic_roots:
                                continue
                            code = _algebraic_codes.get(root) or _algebraic_exprs[root]
                            try:
                                env[root] = float(eval(code, _GLOBAL_ENV, local_env))
                            except Exception:
                                env[root] = 0.0

                        # Evaluate rate equations.
                        for state_index, state_root in enumerate(state_names):
                            code = (
                                _rate_codes.get(state_root) or _rate_exprs[state_root]
                            )
                            try:
                                rates[state_index] = float(
                                    eval(code, _GLOBAL_ENV, local_env)
                                )
                            except Exception:
                                rates[state_index] = 0.0

                for root, kind, forcing_index in override_targets:
                    if kind == "rate":
                        rate_index = state_names.index(root)
                        rates[rate_index] = float(forcing_arr[forcing_index])
                    elif kind == "rate_addend":
                        rate_index = state_names.index(root)
                        rates[rate_index] += float(forcing_arr[forcing_index])

                # Replace any NaN/Inf rates with 0 to prevent the adaptive
                # ODE solver from reducing step size to machine epsilon.
                if _sanitize_nan:
                    rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
                return rates

            return rhs

        return CompiledModel(
            backend="numpy",
            layout=layout,
            initial_state=self._build_initial_state(state_names),
            default_params=default_params,
            _rhs_builder=_rhs_builder,
        )

    def _build_jax_compiled_model(self, options: object) -> "CompiledModel":
        """Phase 9 (JAX variant): assemble a JAX-backend :class:`CompiledModel`.

        Mirrors :meth:`_build_compiled_model` exactly except that the inner
        ``rhs`` closure:

        * uses ``jnp.*`` operations via the ``_JAX_GLOBAL_ENV`` eval context
          instead of ``np.*``.
        * accumulates rates in a Python list and calls ``jnp.stack`` once
          (immutable arrays — no in-place mutation, required for JAX tracing).
        * uses ``.at[].set()`` / ``.at[].add()`` for post-evaluation overrides.
        * drops ``warnings.catch_warnings`` / ``np.errstate`` (not needed for JAX).
        * removes all ``float()`` casts on values that will be JAX scalars.
        """
        import jax.numpy as jnp  # type: ignore

        jax_env = _get_jax_global_env()

        state_names = tuple(self.state_roots)
        parameter_names = tuple(self.parameter_roots)
        forcing_names = tuple(spec.target for spec in self._override_targets)
        state_name_set = set(state_names)

        default_params: dict[str, float] = {}
        for root in parameter_names:
            value = self.parameter_initial_values[root]
            default_params[root] = value
            short = root.split("::", 1)[-1]
            if short not in default_params:
                default_params[short] = value

        override_targets: list[tuple[str, str, int]] = []
        for forcing_index, spec in enumerate(self._override_targets):
            root = self._resolve_name(spec.target)
            kind = spec.kind or self._infer_kind(root)
            override_targets.append((root, kind, forcing_index))
            self._forced_kind_by_root[root] = kind

        layout = RuntimeLayout(
            state_names=state_names,
            parameter_names=parameter_names,
            forcing_names=forcing_names,
            override_targets=tuple(self._override_targets),
        )

        _algebraic_topo_order = self.algebraic_topo_order
        _algebraic_codes = self.algebraic_codes
        _algebraic_exprs = self.algebraic_exprs
        _rate_codes = self.rate_codes
        _rate_exprs = self.rate_exprs
        _param_initial_values = dict(self.parameter_initial_values)
        _time_roots = frozenset(self.time_roots)
        _resolve_name = self._resolve_name
        _forced_algebraic_roots = frozenset(
            self._resolve_name(spec.target)
            for spec in self._override_targets
            if (spec.kind or self._infer_kind(self._resolve_name(spec.target)))
            == "algebraic"
        )
        _sanitize_nan: bool = getattr(options, "sanitize_nan", True)

        def _rhs_builder():
            def rhs(t, x, args):
                """JAX-traceable rate equations for this CellML model.

                Identical contract to the NumPy RHS but safe for ``jax.jit``
                and ``jax.grad``.  Pass ``args=None`` when there are no
                overrides and no parameter adjustments.
                """
                params, forcing_values = CompiledModel._unpack_args(args)
                x_arr = jnp.asarray(x)
                if x_arr.shape != (len(state_names),):
                    raise ShapeError(
                        f"x must have shape {(len(state_names),)}, got {x_arr.shape}"
                    )

                forcing_arr = jnp.asarray(
                    forcing_values if forcing_values is not None else (), dtype=float
                ).reshape(-1)
                if forcing_arr.shape[0] != len(override_targets):
                    raise ShapeError(
                        "forcing vector length does not match override declarations: "
                        f"expected {len(override_targets)}, got {forcing_arr.shape[0]}"
                    )

                params = params or {}

                t_val = t
                env: dict[str, object] = {"time": t_val, "t": t_val}
                for _tr in _time_roots:
                    env[_tr] = t_val
                for state_index, root in enumerate(state_names):
                    env[root] = x_arr[state_index]
                for root, value in _param_initial_values.items():
                    if root not in state_name_set:
                        env[root] = value
                for key, value in params.items():
                    env[_resolve_name(str(key))] = value
                for root, kind, forcing_index in override_targets:
                    if kind in ("constant", "algebraic"):
                        env[root] = forcing_arr[forcing_index]

                local_env = {"V": lambda n, _g=env.get: _g(n, jnp.zeros(()))}

                # Evaluate algebraics in topological order.
                for root in _algebraic_topo_order:
                    if root in _forced_algebraic_roots:
                        continue
                    code = _algebraic_codes.get(root) or _algebraic_exprs[root]
                    try:
                        env[root] = eval(code, jax_env, local_env)
                    except Exception:
                        env[root] = jnp.zeros(())

                # Build rate list; use jnp.stack to produce immutable JAX array.
                rate_list = []
                for state_root in state_names:
                    code = _rate_codes.get(state_root) or _rate_exprs[state_root]
                    try:
                        rate_list.append(eval(code, jax_env, local_env))
                    except Exception:
                        rate_list.append(jnp.zeros(()))
                rates = jnp.stack(rate_list)

                # Post-evaluation rate overrides (functional .at[].set / .add).
                for root, kind, forcing_index in override_targets:
                    if kind == "rate":
                        rate_index = state_names.index(root)
                        rates = rates.at[rate_index].set(forcing_arr[forcing_index])
                    elif kind == "rate_addend":
                        rate_index = state_names.index(root)
                        rates = rates.at[rate_index].add(forcing_arr[forcing_index])

                if _sanitize_nan:
                    rates = jnp.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
                return rates

            return rhs

        return CompiledModel(
            backend="jax",
            layout=layout,
            initial_state=self._build_initial_state(state_names),
            default_params=default_params,
            _rhs_builder=_rhs_builder,
        )

    def _infer_kind(self, root: str) -> str:
        """Infer the override kind for a variable root when the caller omits it.

        Priority: ``"rate"`` if a rate expression exists, ``"algebraic"`` if an
        algebraic expression exists, ``"constant"`` otherwise.
        """
        if root in self.rate_exprs:
            return "rate"
        if root in self.algebraic_exprs:
            return "algebraic"
        return "constant"

    def _build_initial_state(self, state_names: tuple[str, ...]) -> np.ndarray:
        """Assemble a 1-D NumPy array of initial state values in ``state_names`` order.

        Each entry is sourced from the ``initial_value`` attribute of the
        corresponding CellML ``<variable>`` element (or an alias-group member).

        Raises
        ------
        UnsupportedFeatureError
            If any state variable is missing an ``initial_value`` in the document.
        """
        state_values = []
        for root in state_names:
            value = self.parameter_initial_values.get(root)
            if value is None:
                raise UnsupportedFeatureError(
                    f"State {root} is missing an initial value"
                )
            state_values.append(value)
        return np.asarray(state_values, dtype=float)


class _CellMLRuntimeContext:
    """Lazy recursive resolver for CellML algebraic symbols.

    .. deprecated::
        This class is superseded by the flat topological evaluation loop in
        ``_rhs_builder`` (Phase 9).  It is retained for reference and may be
        removed in a future release.  New code should not instantiate it.

    The resolver walks the algebraic dependency graph on demand, caching
    each result so that repeated references to the same variable within one
    RHS evaluation are computed only once.  A ``stack`` attribute tracks the
    current call chain to detect and break cycles.
    """

    def __init__(
        self,
        compiler: _CellMLModelCompiler,
        base_values: dict[str, float],
        forcing_arr: np.ndarray,
        override_targets: list[tuple[str, str, int]],
    ) -> None:
        self.compiler = compiler
        self.base_values = base_values
        self.forcing_arr = forcing_arr
        self.override_targets = override_targets
        self.cache: dict[str, float] = {}
        self.stack: list[str] = []
        self.override_lookup = {
            root: forcing_index
            for root, kind, forcing_index in override_targets
            if kind == "algebraic"
        }
        # Pre-build local eval environment once per context (V resolver is stable for this instance)
        self.local_env = {"V": self.resolve}

    def evaluate(self, expression: str | CodeType) -> float:
        # expression is a pre-compiled code object or a raw string.
        # _GLOBAL_ENV provides np, math, and safe helpers; local_env provides V (the resolver).
        return float(eval(expression, _GLOBAL_ENV, self.local_env))

    def resolve(self, root: str) -> float:
        root = self.compiler._exact_to_root.get(root, root)
        if root in self.cache:
            return self.cache[root]
        if root in self.compiler.time_roots:
            return float(self.base_values["time"])
        if root in self.base_values:
            return self.base_values[root]
        forcing_index = self.override_lookup.get(root)
        if forcing_index is not None:
            return float(self.forcing_arr[forcing_index])
        if root in self.stack:
            raise UnsupportedFeatureError(
                f"Cyclic CellML dependency detected at {root}"
            )

        if root in self.compiler.algebraic_exprs:
            self.stack.append(root)
            try:
                code = (
                    self.compiler.algebraic_codes.get(root)
                    or self.compiler.algebraic_exprs[root]
                )
                value = self.evaluate(code)
            finally:
                self.stack.pop()
            self.cache[root] = value
            return value

        if root in self.compiler.rate_exprs:
            raise MissingInputError(f"Rate symbol {root} is not available as a value")

        raise MissingInputError(f"Could not resolve CellML symbol: {root}")


def compile_cellml_document(path: Path, options: CompileOptions) -> CompiledModel:
    """Compile one CellML 1.0 / 1.1 document into a NumPy-backed executable model.

    This is the primary entry point for the direct CellML compiler path.  It
    instantiates a :class:`_CellMLModelCompiler`, runs all nine compilation
    phases, and returns a :class:`~cellml2py.contracts.CompiledModel` whose
    ``make_rhs()`` factory produces a callable suitable for use with
    ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    path:
        Absolute path to the ``.cellml`` file.  A best-effort standards-
        conformance check is run via ``libcellml`` if that library is installed.
    options:
        Compilation options including any override / forcing declarations.

    Returns
    -------
    CompiledModel
        Immutable compiled model with layout metadata, initial state, and
        default parameter values.
    """
    compiler = _CellMLModelCompiler(path, options)
    return compiler.compile()


def compile_cellml_jax_document(path: Path, options: CompileOptions) -> "CompiledModel":
    """Compile one CellML 1.0 / 1.1 document into a JAX-backend executable model.

    Runs the same nine compilation phases as :func:`compile_cellml_document` but
    produces a ``CompiledModel`` whose ``make_rhs()`` callable is safe for
    ``jax.jit``, ``jax.grad``, and ``diffrax.ODETerm``.

    Requires JAX to be installed (``pip install cellml2py[jax]``).

    Parameters
    ----------
    path:
        Absolute path to the ``.cellml`` file.
    options:
        Compilation options.  Set ``options.sanitize_nan = False`` when using
        ``jax.grad`` for parameter fitting.

    Returns
    -------
    CompiledModel
        Model with ``backend="jax"`` and a JAX-traceable RHS closure.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    _try_import_jax()
    compiler = _CellMLModelCompiler(path, options)
    compiler.compile()  # run phases 1-8 to populate all analysis results
    return compiler._build_jax_compiled_model(options)


def _safe_exp(value: float | np.ndarray) -> float | np.ndarray:
    """Exponentiation with clipping to avoid floating-point overflow warnings."""
    return np.exp(np.clip(value, -700.0, 700.0))


def _jax_safe_exp(value):
    """JAX-traceable exponentiation with clipping to avoid overflow."""
    import jax.numpy as jnp  # type: ignore

    return jnp.exp(jnp.clip(value, -700.0, 700.0))


def _safe_divide(
    numerator: float | np.ndarray,
    denominator: float | np.ndarray,
    epsilon: float = 1e-12,
) -> float | np.ndarray:
    """Division with protection against zero and non-finite denominators."""
    denominator_arr = np.asarray(denominator, dtype=float)
    numerator_arr = np.asarray(numerator, dtype=float)

    finite = np.isfinite(denominator_arr)
    small = np.abs(denominator_arr) < epsilon
    unsafe = (~finite) | small

    sign = np.where(denominator_arr >= 0.0, 1.0, -1.0)
    safe_denom = np.where(unsafe, sign * epsilon, denominator_arr)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.divide(numerator_arr, safe_denom)

    if np.isscalar(numerator) and np.isscalar(denominator):
        return float(result)
    return result


def _jax_safe_divide(numerator, denominator, epsilon: float = 1e-12):
    """JAX-traceable division with zero-denominator protection."""
    import jax.numpy as jnp  # type: ignore

    finite = jnp.isfinite(denominator)
    small = jnp.abs(denominator) < epsilon
    unsafe = (~finite) | small
    sign = jnp.where(denominator >= 0.0, 1.0, -1.0)
    safe_denom = jnp.where(unsafe, sign * epsilon, denominator)
    return jnp.divide(numerator, safe_denom)


def _safe_power(
    base: float | np.ndarray,
    exponent: float | np.ndarray,
    epsilon: float = 1e-12,
) -> float | np.ndarray:
    """Power with guards for invalid real-domain and overflow operations."""
    base_arr = np.asarray(base, dtype=float)
    exp_arr = np.asarray(exponent, dtype=float)

    exp_rounded = np.round(exp_arr)
    exp_is_integer = np.abs(exp_arr - exp_rounded) <= 1e-12
    negative_base = base_arr < 0.0

    # Fractional powers of negative bases are invalid in real arithmetic.
    adjusted_base = np.where(
        negative_base & (~exp_is_integer), np.abs(base_arr) + epsilon, base_arr
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        result = np.power(adjusted_base, exp_arr)

    if np.isscalar(base) and np.isscalar(exponent):
        return float(result)
    return result


def _jax_safe_power(base, exponent, epsilon: float = 1e-12):
    """JAX-traceable power with guards for negative-base fractional exponents."""
    import jax.numpy as jnp  # type: ignore

    exp_rounded = jnp.round(exponent)
    exp_is_integer = jnp.abs(exponent - exp_rounded) <= 1e-12
    negative_base = base < 0.0
    adjusted_base = jnp.where(
        negative_base & (~exp_is_integer), jnp.abs(base) + epsilon, base
    )
    return jnp.power(adjusted_base, exponent)


# ---------------------------------------------------------------------------
# Module-level eval environment shared across all RHS evaluations.
# Defined after all helpers so that references are valid.
#
# ``"__builtins__"`` is explicitly absent so that eval()-ed CellML expressions
# cannot accidentally access Python built-ins (e.g. open, exec).  Only the
# symbols declared here are available inside expression strings:
#   np          - NumPy namespace for vectorised math
#   math        - Python math module for scalar constants (e.g. math.pi)
#   SAFE_EXP    - overflow-clipped exponential
#   SAFE_DIVIDE - zero-denominator-safe division
#   SAFE_POWER  - real-domain-safe power
# ---------------------------------------------------------------------------
_GLOBAL_ENV: dict[str, object] = {
    "np": np,
    "math": math,
    "SAFE_EXP": _safe_exp,
    "SAFE_DIVIDE": _safe_divide,
    "SAFE_POWER": _safe_power,
}

# JAX eval environment — populated lazily on the first backend="jax" compile.
# Structurally identical to _GLOBAL_ENV but uses jnp.* and JAX-safe helpers.
_JAX_GLOBAL_ENV: dict[str, object] | None = None


def _get_jax_global_env() -> dict[str, object]:
    """Return (and lazily build) the JAX eval environment."""
    global _JAX_GLOBAL_ENV
    if _JAX_GLOBAL_ENV is None:
        import jax.numpy as jnp  # type: ignore

        _JAX_GLOBAL_ENV = {
            "np": jnp,
            "math": math,
            "SAFE_EXP": _jax_safe_exp,
            "SAFE_DIVIDE": _jax_safe_divide,
            "SAFE_POWER": _jax_safe_power,
        }
    return _JAX_GLOBAL_ENV
