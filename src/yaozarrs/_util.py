import re
import warnings
from functools import partial
from typing import Annotated, TypeAlias

from coverage.python import os
from pydantic import AfterValidator, BeforeValidator
from traitlets.traitlets import Any


def warn_if_risky_node_name(path: str, field_name: str = "") -> str:
    """Warn if the given Zarr node name is potentially risky.

    "risky" names include characters outside of the set [A-Za-z0-9._-], which may
    cause issues on some filesystems or when used in URLs.

    set YAOZARRS_ALLOW_RISKY_NODE_NAMES=1 to opt out of this warning.
    """
    risky_chars = re.findall(r"[^A-Za-z0-9._-]", path)
    if risky_chars and not os.getenv("YAOZARRS_ALLOW_RISKY_NODE_NAMES"):
        if field_name:
            for_field = f" on field '{field_name}'"
        else:
            for_field = ""
        warnings.warn(
            f"The name {path!r}{for_field} contains potentially risky characters when "
            f"used as a zarr node: {set(risky_chars)}.\nConsider using only "
            "alphanumeric characters, dots (.), underscores (_), or hyphens (-) to "
            "avoid issues on some filesystems or when used in URLs. "
            "Set YAOZARRS_ALLOW_RISKY_NODE_NAMES=1 to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
    return path


SuggestDatasetPath = AfterValidator(
    partial(warn_if_risky_node_name, field_name="Dataset.path")
)


def _warn_non_spec_fov_name(path: Any) -> str:
    """Warning validator for FOV names.

    The NGFF spec states that FOV names should be alphanumeric only: [A-Za-z0-9].
    This is overly restrictive, so we allow a relaxed set of characters [A-Za-z0-9._-]
    but warn the user if they use characters outside of the spec.

    set YAOZARRS_STRICT_FOV_NAMES=1 to enforce strict compliance.
    set YAOZARRS_IGNORE_RISKY_FOV_NAMES=1 to suppress this warning.
    """
    _path = str(path)
    if os.getenv("YAOZARRS_STRICT_FOV_NAMES"):
        strict_pattern = r"^[A-Za-z0-9]+$"
        if not re.match(strict_pattern, _path):
            raise ValueError(f"String should match pattern {strict_pattern}.")

    relaxed_pattern = r"^[A-Za-z0-9._-]+$"
    if not re.match(relaxed_pattern, _path):
        raise ValueError(f"String should match pattern {relaxed_pattern}.")

    risky_chars = re.findall(r"[^A-Za-z0-9]", _path)
    if risky_chars and not os.getenv("YAOZARRS_IGNORE_RISKY_FOV_NAMES"):
        warnings.warn(
            f"The FieldOfView.path {_path!r} contains characters outside of the NGFF "
            f"spec ([A-Za-z0-9]): {set(risky_chars)}.\nWhile yaozarrs DOES allow these "
            "characters, they are not strictly spec-compliant and may cause "
            "compatibility issues with strict NGFF-compliant tools or libraries.\n"
            "See https://github.com/ome/ngff-spec/pull/71.",
            UserWarning,
            stacklevel=3,
        )

    return _path


RelaxedFOVPathName: TypeAlias = Annotated[str, BeforeValidator(_warn_non_spec_fov_name)]
