"""Custom exceptions for the cellml2py package.

All exceptions derive from :class:`CellML2PyError` so callers can catch any
package error with a single except clause.
"""


class CellML2PyError(Exception):
    """Base exception for all cellml2py errors."""


class UnsupportedFeatureError(CellML2PyError):
    """Raised when the CellML document uses a construct not yet implemented.

    Examples include MathML operators outside the supported subset, implicit
    algebraic equations, or CellML 2.0 structural elements.
    """


class MissingInputError(CellML2PyError):
    """Raised when a required forcing value or parameter name is absent.

    Typical causes: a variable name supplied to ``params`` that does not match
    any canonical or short name in the compiled model, or a forcing vector
    whose length does not match the number of declared override targets.
    """


class ShapeError(CellML2PyError):
    """Raised when an input array has an unexpected shape.

    Occurs when the state vector ``x`` passed to ``rhs()`` has a different
    number of elements than ``len(layout.state_names)``.
    """
