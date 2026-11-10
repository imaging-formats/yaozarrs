"""Checking that a writer is given its own version's models."""

from __future__ import annotations

from typing import Any


def _model_name(cls: type) -> str:
    # e.g. "v05.Image" for yaozarrs.v05._image.Image
    parts = cls.__module__.split(".")
    if parts[0] == "yaozarrs" and len(parts) > 1:
        return f"{parts[1]}.{cls.__qualname__}"
    return cls.__qualname__


def _check_model(obj: Any, expected: type, context: str = "") -> None:
    """Raise TypeError unless `obj` is an instance of this writer's `expected` model.

    (e.g. a v05.Image passed to the v06 writer would otherwise be written as-is)
    """
    if not isinstance(obj, expected):
        raise TypeError(
            f"{context}expected a {_model_name(expected)}, got {_model_name(type(obj))}"
        )
