"""Portable path references for persisted experiment evidence."""

from __future__ import annotations

import os
from pathlib import Path


def relative_posix_reference(
    path: str | os.PathLike[str],
    *,
    base: str | os.PathLike[str],
) -> str:
    """Return ``path`` relative to ``base`` using POSIX separators.

    Inputs are resolved to absolute paths before relativization. This makes
    references independent of the caller's working directory while ensuring
    that an impossible cross-drive reference fails instead of silently
    persisting a machine-specific absolute path.
    """

    target = Path(path).resolve(strict=False)
    anchor = Path(base).resolve(strict=False)
    try:
        relative = os.path.relpath(target, start=anchor)
    except ValueError as error:
        raise ValueError(
            "cannot create a portable relative reference across filesystem drives: "
            f"{target} relative to {anchor}"
        ) from error
    reference = Path(relative).as_posix()
    if Path(reference).is_absolute():
        raise ValueError("portable path reference unexpectedly remained absolute")
    return reference


__all__ = ["relative_posix_reference"]
