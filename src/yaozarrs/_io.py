import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

if TYPE_CHECKING:
    import io

    import fsspec
else:
    try:
        import fsspec
    except ImportError:
        fsspec = None


F = TypeVar("F", bound=Callable[..., object])


def _require_fsspec(func: F) -> F:
    """Decorator to ensure fsspec is available for functions that need it."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if fsspec is None:
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
    # Determine the target JSON file URI
    if uri_str.endswith((".json", ".zattrs")):
        json_uri = uri_str
    else:
        attrs_file: str = "zarr.json"  # TODO find the right one...
        # Assume it's a directory, look for zarr attributes file inside it
        json_uri = f"{uri_str.rstrip('/')}/{attrs_file}"

    # Load JSON content using fsspec
    try:
        with fsspec.open(json_uri, "r") as f:
            json_content = cast("io.TextIOBase", f).read()

    except Exception as e:
        msg = f"Could not load JSON from URI: {json_uri}:\n{e}"
        raise FileNotFoundError(msg) from e

    return json_content, uri_str
