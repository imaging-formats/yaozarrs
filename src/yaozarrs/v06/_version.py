"""Shared version type for OME-NGFF v0.6 models."""

from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator


def _validate_v06_version(v: str) -> str:
    # Accept the 0.6 line: "0.6", "0.6.0", and dev tags like "0.6.dev4". Reject a
    # 0.6.Z *patch* release where Z is a numeral > 0 (e.g. "0.6.1") -- that would
    # be different content we don't claim to support -- and anything not in 0.6
    # (e.g. "0.5", "0.66").
    parts = v.split(".")
    ok = len(parts) >= 2 and parts[0] == "0" and parts[1] == "6"
    if ok and len(parts) >= 3 and parts[2].isdigit() and int(parts[2]) > 0:
        ok = False
    if not ok:
        raise ValueError(
            f"version must be '0.6', '0.6.0', or a '0.6.dev*' tag, got {v!r}"
        )
    return v


# `Literal["0.6"]` surfaces the canonical value for IDE autocompletion, while the
# `| str` + validator accept any `"0.6*"` string (e.g. "0.6.dev4", "0.6", "0.6.1")
# so we don't have to bump anything when v0.6 lands or when testing newer dev tags.
OMEV06: TypeAlias = Annotated[
    Literal["0.6"] | str, AfterValidator(_validate_v06_version)
]
