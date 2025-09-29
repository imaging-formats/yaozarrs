"""Storage validation for OME-ZARR v0.5 hierarchies.

This module provides functions to validate that OME-ZARR v0.5 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.

Requires zarr to be installed for full validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import zarr
from typing_extensions import NotRequired

from yaozarrs._validate import validate_ome_object

from ._zarr_json import OMEAttributes, OMEZarrGroupJSON

if TYPE_CHECKING:
    from zarr.storage import StoreLike


class ErrorDetails(TypedDict):
    type: str
    """
    The type of error that occurred, this is an identifier designed for
    programmatic use that will change rarely or never.

    `type` is unique for each error message, and can hence be used as an identifier to
    build custom error messages.
    """
    loc: tuple[int | str, ...]
    """Tuple of strings and ints identifying where in the schema the error occurred."""
    msg: str
    """A human readable error message."""
    input: Any
    """The input data at this `loc` that caused the error."""
    ctx: NotRequired[dict[str, Any]]
    """
    Values which are required to render the error message, and could hence be useful in
    rendering custom error messages.
    Also useful for passing custom error data forward.
    """
    url: NotRequired[str]
    """A URL giving information about the error."""


class StorageValidationError(ValueError):
    """`StorageValidationError` is raised when validation of zarr storage fails.

    It contains a list of errors which detail why validation failed.
    """

    @property
    def title(self) -> str:
        """The title of the error, as used in the heading of `str(validation_error)`."""
        return "StorageValidationError"

    def errors(
        self,
        *,
        include_url: bool = True,
        include_context: bool = True,
        include_input: bool = True,
    ) -> list[ErrorDetails]:
        """
        Details about each error in the validation error.

        Parameters
        ----------
        include_url: bool
            Whether to include a URL to documentation on the error each error.
        include_context: bool
            Whether to include the context of each error.
        include_input: bool
            Whether to include the input value of each error.

        Returns
        -------
            A list of `ErrorDetails` for each error in the validation error.
        """
        return []


def _get_zarr_group(uri: str | StoreLike | OMEZarrGroupJSON) -> zarr.Group:
    try:
        import zarr
    except ImportError as e:
        raise ImportError(
            "zarr must be installed to use storage validation. "
            "Please install yaozarrs with the `storage` extra, "
            "e.g. `pip install yaozarrs[storage]`."
        ) from e

    if isinstance(uri, (str, Path)):
        uri = str(os.path.expanduser(uri))
    return zarr.open_group(uri)


def validate_zarr_store(obj: OMEZarrGroupJSON | zarr.Group | StoreLike):
    """Validate an OME-Zarr v0.5 storage structure.

    Raises
    ------
    StorageValidationError
        If the storage structure is invalid.
    ImportError
        If zarr is not installed.
    """
    attrs_model: OMEAttributes | None = None
    if isinstance(obj, zarr.Group):
        zarr_group = obj
    elif isinstance(obj, OMEZarrGroupJSON):
        attrs_model = obj.attributes
        zarr_group = _get_zarr_group(obj.uri)
    else:
        zarr_group = _get_zarr_group(obj)

    _validate_zarr_group(zarr_group, attrs_model)


def _validate_zarr_group(zarr_group: zarr.Group, attrs_model: OMEAttributes | None):
    if attrs_model is None:
        # extract the model from the zarr attributes
        attrs = zarr_group.attrs.asdict()
        attrs_model = validate_ome_object(attrs, OMEAttributes)

    # at this point we have a valid zarr_group, and OMEAttributes model
    # (which means the "ome" key is present and valid)
    # and we can begin to validate that the storage structure itself is valid,
    # by traversing the structure recursively
