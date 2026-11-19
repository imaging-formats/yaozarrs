import os
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    import fsspec
    import fsspec.core
else:
    try:
        import fsspec
        import fsspec.core
    except ImportError:
        fsspec = None


F = TypeVar("F", bound=Callable[..., object])

_METADATA_FILES = ("zarr.json", ".zattrs")
# the errors that mean a key is missing (as in fsspec's FSMap.missing_exceptions)
_MISSING = (FileNotFoundError, IsADirectoryError, NotADirectoryError)


def _require_fsspec(func: F) -> F:
    """Decorator to ensure fsspec is available for functions that need it."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if fsspec is None:  # pragma: no cover
            msg = (
                f"fsspec is required for {func.__name__!r}.\n"
                "Install with: 'pip install yaozarrs[io]' or 'pip install fsspec'"
            )
            raise ImportError(msg)
        return func(*args, **kwargs)

    return cast("F", wrapper)


@_require_fsspec
def read_json_from_uri(uri: str | os.PathLike) -> tuple[str, str]:
    """Read JSON content from a URI (local or remote) using fsspec.

    Parameters
    ----------
    uri : str or os.PathLike
        The URI to read the JSON data from.  This can be a local file path,
        or a remote URL (e.g. s3://bucket/key/some_file.zarr).  It can be a zarr
        group directory, or a direct path to a JSON file (e.g. zarr.json or
        .zattrs) inside a zarr group.

    Returns
    -------
    tuple[str, str]
        A tuple containing the JSON content as a string, and the normalized URI string.
    """
    uri_str = os.fspath(uri)
    if uri_str.endswith(_METADATA_FILES):
        candidates = [uri_str]
    else:  # a zarr group directory
        uri_str = uri_str.rstrip("/") + "/"
        candidates = [uri_str + name for name in _METADATA_FILES]

    # (`cat_file` raises FileNotFoundError only for a missing file. Not `fs.exists()`
    # or `fsspec.open()`: over http they report *any* error, e.g. a refused
    # connection, as a missing file)
    for json_uri in candidates:
        fs, path = fsspec.core.url_to_fs(json_uri)
        try:
            return fs.cat_file(path).decode("utf-8"), json_uri
        except _MISSING:
            continue
    raise FileNotFoundError(f"Could not find zarr group metadata in: {uri_str}")
