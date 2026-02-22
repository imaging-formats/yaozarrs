import os
import re
import warnings
from functools import partial
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, BeforeValidator

BAD_NODE_RE = re.compile(r"^(?:|\.+|__.*|.*\/.*)$")
"""Regular expression matching invalid Zarr node names.

- must not be the empty string ("")
- must not include the character "/"
- must not be a string composed only of period characters, e.g. "." or ".."
- must not start with the reserved prefix "__"
"""


def validate_node_name(
    path: str, field_name: str = "", allow_sep: str | None = "/"
) -> str:
    """Warn if the given Zarr node name is potentially risky.

    "risky" names include characters outside of the set [A-Za-z0-9._-/], which may
    cause issues on some filesystems or when used in URLs.

    set YAOZARRS_ALLOW_RISKY_NODE_NAMES=1 to opt out of this warning.

    Parameters
    ----------
    path : str
        The Zarr node name to validate.
    field_name : str, optional
        The name of the field being validated (for warning messages), by default ""
    allow_sep : str | None, optional
        If provided, allows the path to *include* this separator character, in which
        case parts will be validated separately. Use when you want to allow a string
        to represent a nested path within a Zarr store, by default "/".
    """
    if allow_sep:
        # split on allow_sep to allow nested paths
        parts = path.split(allow_sep)
    else:
        parts = [path]
    for part in parts:
        if BAD_NODE_RE.match(part):
            raise ValueError(
                f"The name {path!r} is not a valid Zarr node name. See "
                "https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#node-names"
            )

        # note, we allow '/' here to support nested paths within a Zarr store
        # using logical paths rather than file system paths.
        risky_chars = re.findall(r"[^A-Za-z0-9._-]", part)
        if risky_chars and not os.getenv("YAOZARRS_ALLOW_RISKY_NODE_NAMES"):
            for_field = f" on field '{field_name}'" if field_name else ""
            warnings.warn(
                f"The name {part!r}{for_field} contains potentially risky characters "
                f"when used as a zarr node: {set(risky_chars)}.\nConsider using only "
                "alphanumeric characters, dots (.), underscores (_), or hyphens (-) to "
                "avoid issues on some filesystems or when used in URLs. "
                "Set YAOZARRS_ALLOW_RISKY_NODE_NAMES=1 to suppress this warning.",
                UserWarning,
                stacklevel=3,
            )
    return path


SuggestDatasetPath = AfterValidator(
    partial(validate_node_name, field_name="Dataset.path")
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
