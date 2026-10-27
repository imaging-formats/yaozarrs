import warnings
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator

from yaozarrs._validation_warning import ValidationWarning


def _warn_prerelease(v: str) -> str:
    # The schema also accepts the release candidate "0.6rc0" (the prose implies "0.6").
    if v != "0.6":
        warnings.warn(
            f"OME-Zarr version {v!r} is a pre-release of '0.6'.",
            ValidationWarning,
            stacklevel=2,
        )
    return v


VersionStr: TypeAlias = Annotated[
    Literal["0.6", "0.6rc0"], AfterValidator(_warn_prerelease)
]
