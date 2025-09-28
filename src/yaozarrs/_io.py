import os
from collections.abc import Iterable
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

if TYPE_CHECKING:
    import io

    import fsspec
    import fsspec.utils
else:
    try:
        import fsspec
        import fsspec.utils
    except ImportError:
        fsspec = None


F = TypeVar("F", bound=Callable[..., object])


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
    json_uri = _find_zarr_group_metadata(uri_str)

    # Load JSON content using fsspec
    try:
        with fsspec.open(json_uri, "r") as f:
            json_content = cast("io.TextIOBase", f).read()

    except FileNotFoundError as e:
        msg = f"Could not load JSON from URI: {json_uri}:\n{e}"
        raise FileNotFoundError(msg) from e

    return json_content, json_uri


def _find_zarr_group_metadata(
    uri_str: str, candidates: Iterable[str] = ("zarr.json", ".zattrs")
) -> str:
    """Return path to zarr group metadata file inside a zarr group directory."""
    # If the URI already points to a known metadata file, return it directly
    if uri_str.endswith(("zarr.json", ".zattrs")):
        return uri_str

    # we assume it's a zarr group directory
    # we now need to use fsspec to use the filesystem
    # (which may be local or remote)
    # to find either zarr.json or .zattrs
    options = fsspec.utils.infer_storage_options(uri_str)
    protocol = options.get("protocol", "file")
    fs = cast("fsspec.AbstractFileSystem", fsspec.filesystem(protocol))

    for candidate in candidates:
        json_uri = uri_str + fs.sep + candidate
        if fs.exists(json_uri):
            return json_uri

    raise FileNotFoundError(f"Could not find zarr group metadata in: {uri_str}")
