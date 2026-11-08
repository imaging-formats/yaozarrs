"""Types used in the signatures of yaozarrs' public API.

Use these to annotate values that you pass to yaozarrs, or get back from it.
They exist only for type checkers, so import them under `if TYPE_CHECKING:`
(with `from __future__ import annotations`, or quoted annotations, on Python < 3.14):

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yaozarrs.typing import CompressionName

compression: CompressionName = "zstd"
```

Since that import never runs, it works with any version of yaozarrs at runtime
(including those that predate this module), so it needs no minimum version.

Removing or renaming a name here is a breaking change.  A `Literal` may gain values
(e.g. a new compression codec) in a minor release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal, TypeAlias

    import tensorstore
    import zarr
    from numpy.typing import DTypeLike

    # (classes and runtime aliases that appear in public signatures: defined
    # elsewhere, where yaozarrs uses them at runtime)
    from yaozarrs._errors import ErrorDetails as ErrorDetails
    from yaozarrs._validate import AnyOME as AnyOME
    from yaozarrs._validate import AnyOMEGroup as AnyOMEGroup
    from yaozarrs._zarr import ZarrArray as ZarrArray
    from yaozarrs._zarr import ZarrMetadata as ZarrMetadata
    from yaozarrs.write._common._arrays import CreateArrayFunc as CreateArrayFunc

    __all__ = [  # noqa: RUF022
        # writing
        "CompressionName",
        "WriterName",
        "ZarrWriter",
        "CreateArrayFunc",
        "AnyZarrArray",
        "ArrayLike",
        "ArrayOrPyramid",
        "ShapeLike",
        "ShapeAndDType",
        "ShapeAndDTypeOrPyramid",
        # reading and validating
        "AnyOME",
        "AnyOMEGroup",
        "ZarrArray",
        "ZarrMetadata",
        "ErrorDetails",
    ]

    CompressionName: TypeAlias = Literal["blosc-zstd", "blosc-lz4", "zstd", "none"]
    """Compression codec for the arrays that yaozarrs writes."""

    WriterName: TypeAlias = Literal["zarr", "tensorstore", "auto"]
    """Built-in array backend ("auto" prefers tensorstore, then zarr-python)."""

    ZarrWriter: TypeAlias = WriterName | CreateArrayFunc
    """Array backend: a built-in one, or a custom array creation function."""

    AnyZarrArray: TypeAlias = zarr.Array[Any] | tensorstore.TensorStore
    """Array created by a built-in backend."""

    ArrayLike: TypeAlias = Any
    """Array of data to write (e.g. a numpy or dask array)."""

    ArrayOrPyramid: TypeAlias = ArrayLike | Sequence[ArrayLike]
    """One array, or a sequence of them (one per dataset of a multiscale)."""

    ShapeLike: TypeAlias = tuple[int, ...]
    """Shape of an array, or of its chunks or shards."""

    ShapeAndDType: TypeAlias = tuple[ShapeLike, DTypeLike]
    """A `(shape, dtype)` tuple describing an array to create."""

    ShapeAndDTypeOrPyramid: TypeAlias = ShapeAndDType | Sequence[ShapeAndDType]
    """One `(shape, dtype)` tuple, or a sequence of them (one per dataset)."""

else:
    # (none of the names above exist at runtime, so a star import gets none of them)
    __all__: list[str] = []
    # (`__all__` above, as a runtime set: kept in sync by tests/test_public_typing.py)
    _TYPE_CHECKING_ONLY = {
        "AnyOME",
        "AnyOMEGroup",
        "AnyZarrArray",
        "ArrayLike",
        "ArrayOrPyramid",
        "CompressionName",
        "CreateArrayFunc",
        "ErrorDetails",
        "ShapeAndDType",
        "ShapeAndDTypeOrPyramid",
        "ShapeLike",
        "WriterName",
        "ZarrArray",
        "ZarrMetadata",
        "ZarrWriter",
    }

    # (hidden from type checkers, or any misspelled name would type-check)
    def __getattr__(name: str) -> NoReturn:
        if name not in _TYPE_CHECKING_ONLY:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        raise ImportError(
            f"{name!r} cannot be imported from {__name__!r} at runtime: the names in "
            f"{__name__!r} exist only for type checkers.  Import it under "
            "`if TYPE_CHECKING:` instead.",
            name=__name__,
        )
