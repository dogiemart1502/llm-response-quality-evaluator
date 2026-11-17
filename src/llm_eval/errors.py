"""Exception hierarchy used throughout the package.

All errors raised deliberately by this package derive from
:class:`EvaluatorError`, so callers (including the CLI) can catch a single type
and present a clean message instead of a traceback.
"""

from __future__ import annotations


class EvaluatorError(Exception):
    """Base class for all expected, user-facing errors."""


class ValidationError(EvaluatorError, ValueError):
    """Raised when input data does not match the expected schema.

    Args:
        message: Human-readable description of the problem.
        path: Location of the offending value, e.g. ``"items[2].prompt"``.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class InputFileError(EvaluatorError, OSError):
    """Raised when an input file is missing, unreadable, or not valid JSON."""
