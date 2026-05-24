"""Logging configuration for pydefect-complex.

Usage::

    from pydefect_complex import configure_logging
    configure_logging()                         # INFO to stderr
    configure_logging(verbose=True)             # DEBUG to stderr
    configure_logging(log_file="pipeline.log")  # also write to file
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_configured: bool = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger for a pydefect-complex submodule.

    All internal modules call this instead of
    ``logging.getLogger(__name__)`` so the handler chain is
    guaranteed available once the user calls ``configure_logging()``.
    """
    logger = logging.getLogger(name)
    if not _configured:
        _ensure_has_handler(logger)
    return logger


def _ensure_has_handler(logger: logging.Logger) -> None:
    """Attach a NullHandler if no handler is reachable."""
    if not logger.handlers:
        # Walk up the parent chain
        cur: logging.Logger | None = logger
        has_handler = False
        while cur is not None:
            if cur.handlers:
                has_handler = True
                break
            cur = cur.parent
        if not has_handler:
            logger.addHandler(logging.NullHandler())


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = False,
    quiet_third_party: bool = True,
) -> None:
    """Configure the ``pydefect_complex`` root logger.

    Call once at the start of a script.  Without this call, internal
    log messages are silently dropped (NullHandler).

    Args:
        level: Console handler level (default INFO).
        log_file: If given, also write DEBUG-level output to this file.
        verbose: If True, set console level to DEBUG.
        quiet_third_party: If True (default), suppress noisy INFO
            messages from pydefect, vise, and pymatgen.
    """
    global _configured

    root = logging.getLogger("pydefect_complex")
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Remove any previously attached handlers to avoid duplicates
    root.handlers.clear()

    # --- console (stderr) ---
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # --- file (optional) ---
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

    # --- squelch third-party noise ---
    if quiet_third_party:
        for name in ("vise", "pydefect", "pymatgen"):
            logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True