"""CellML intake and backend dispatch.

This module is a thin adapter between the public API (:mod:`cellml2py.api`)
and the direct CellML compiler (:mod:`cellml2py.cellml_compiler`).  Its
primary responsibility is to validate the requested backend and delegate to
:func:`~cellml2py.cellml_compiler.compile_cellml_document`.

Adding a new backend (e.g. JAX) requires only extending the validation check
in :func:`compile_cellml_model` and providing a parallel entry point in
:mod:`cellml2py.cellml_compiler`.
"""

from __future__ import annotations

from pathlib import Path

from .cellml_compiler import compile_cellml_document
from .contracts import CompileOptions, CompiledModel


def compile_cellml_model(
    path: Path,
    backend: str,
    options: CompileOptions,
) -> CompiledModel:
    """Compile a CellML file directly into a runtime RHS callable.

    Parameters
    ----------
    path:
        Absolute path to the ``.cellml`` file.
    backend:
        Numerical backend identifier.  Only ``"numpy"`` is supported;
        passing any other value raises ``ValueError``.
    options:
        Compilation options including override target declarations.

    Returns
    -------
    CompiledModel
        Compiled model produced by
        :func:`~cellml2py.cellml_compiler.compile_cellml_document`.

    Raises
    ------
    ValueError
        If *backend* is not ``"numpy"`` or ``"jax"``.
    """

    if backend == "jax":
        from .cellml_compiler import compile_cellml_jax_document

        return compile_cellml_jax_document(path, options)
    if backend != "numpy":
        raise ValueError(
            f"Unknown backend {backend!r}.  Choose 'numpy' or 'jax'."
        )
    return compile_cellml_document(path, options)