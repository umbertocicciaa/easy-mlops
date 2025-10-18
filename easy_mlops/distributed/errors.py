"""Error handling utilities for the distributed master service."""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional


def safe_call(func, *args, **kwargs) -> Any:
    """Execute a callable and return (result, error)."""
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        tb = traceback.format_exc()
        return None, (exc, tb)


def format_error(exc: Exception, tb: str) -> Dict[str, str]:
    """Create a JSON-serializable payload describing an exception."""
    return {
        "message": str(exc),
        "type": exc.__class__.__name__,
        "traceback": tb,
    }
