"""Shared version type for OME-NGFF v0.6 models."""

import re
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator

# The current stable spec target: the "0.6rc0" release-candidate tag.
CURRENT_VERSION = "0.6rc0"

# Accept the whole 0.6 line: "0.6", "0.6.0", "0.6rcN" (release candidates), and
# dev tags like "0.6.dev4" (older pre-rc documents, for backwards parsing).
# Reject a 0.6.Z *patch* release where Z is a numeral > 0 (e.g. "0.6.1") -- that
# would be different content we don't claim to support -- and anything not in
# 0.6 (e.g. "0.5", "0.66").
_V06_PATTERN = re.compile(r"^0\.6(\.0|\.dev\d+|rc\d+)?$")


def _validate_v06_version(v: str) -> str:
    if not _V06_PATTERN.fullmatch(v):
        raise ValueError(
            f"version must be '0.6', '0.6.0', a '0.6rcN' tag, or a "
            f"'0.6.devN' tag, got {v!r}"
        )
    return v


# `Literal["0.6rc0"]` surfaces the canonical value for IDE autocompletion, while
# the `| str` + validator accept the full 0.6 line (e.g. "0.6.dev4", "0.6",
# "0.6.0", "0.6rc1") so we don't have to bump anything for a new rc/dev tag.
OMEV06: TypeAlias = Annotated[
    Literal["0.6rc0"] | str, AfterValidator(_validate_v06_version)
]
