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
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if fsspec is None:
            msg = (
                f"fsspec is required for {func.__name__!r}.\n"
                "Install with: pip install yaozarrs[io]"
            )
            raise ImportError(msg)
        return func(*args, **kwargs)

    return cast("F", wrapper)


@_require_fsspec
def read_json_from_uri(uri: str | os.PathLike, attrs_file: str = "zarr.json") -> str:
    uri_str = os.fspath(uri)

    # Determine the target JSON file URI
    if uri_str.endswith((".json", ".zattrs")):
        json_uri = uri
    else:
        # Assume it's a directory, look for zarr attributes file inside it
        json_uri = f"{uri_str.rstrip('/')}/{attrs_file}"

    # Load JSON content using fsspec
    try:
        with fsspec.open(json_uri, "r") as f:
            json_content = cast("io.TextIOBase", f).read()

    except Exception as e:
        msg = f"Could not load JSON from URI: {json_uri}"
        raise FileNotFoundError(msg) from e

    return json_content
