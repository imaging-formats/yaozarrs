from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from typing_extensions import NotRequired, TypedDict

from yaozarrs._zarr import ZarrGroup, open_group

if TYPE_CHECKING:
    from pathlib import Path


def validate_zarr_store(obj: ZarrGroup | str | Path | Any) -> None:
    """Validate an OME-Zarr v0.5 storage structure.

    Parameters
    ----------
    obj : OMEZarrGroupJSON | ZarrGroup | str | Path | Any
        The zarr store to validate. Can be a URI string, a Path, a parsed
        OMEZarrGroupJSON object, a ZarrGroup instance, or a zarr.Group object
        (for backwards compatibility).

    Raises
    ------
    StorageValidationError
        If the storage structure is invalid.
    """
    zarr_group = open_group(obj)
    ome_version = zarr_group.ome_version()
    if ome_version == "0.5":
        from yaozarrs.v05._storage import StorageValidatorV05

        # Validate the storage structure using the visitor pattern
        result = StorageValidatorV05.validate_group(zarr_group)
    else:
        raise NotImplementedError(
            f"Structural validation for OME-Zarr version {ome_version} is "
            "not implemented."
        )

    # Raise error if any validation issues found
    if not result.is_valid:
        raise StorageValidationError(result.errors)


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

    def __init__(self, errors: list[ErrorDetails]) -> None:
        self._errors = errors
        super().__init__(self._error_message())

    def _error_message(self) -> str:
        """Generate a readable error message from all validation errors."""
        if not self._errors:  # pragma: no cover
            return "No validation errors"

        lines = [f"{len(self._errors)} validation error(s) for storage structure:"]
        for i, error in enumerate(self._errors, 1):
            lines.append(
                f"{i:>2}. {error['msg']} (type={error['type']}, loc={error['loc']})"
            )
        return "\n".join(lines)

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
        filtered_errors: list[ErrorDetails] = []
        for error in self._errors:
            filtered_error = {
                "type": error["type"],
                "loc": error["loc"],
                "msg": error["msg"],
            }
            if include_input and "input" in error:
                filtered_error["input"] = error["input"]
            if include_context and "ctx" in error:  # pragma: no cover
                filtered_error["ctx"] = error["ctx"]
            if include_url and "url" in error:  # pragma: no cover
                filtered_error["url"] = error["url"]
            filtered_errors.append(filtered_error)
        return filtered_errors


class StorageErrorType(Enum):
    bf2raw_invalid_image = auto()
    bf2raw_no_images = auto()
    bf2raw_path_not_group = auto()
    dataset_dimension_mismatch = auto()
    dataset_not_array = auto()
    dataset_path_not_found = auto()
    dimension_names_mismatch = auto()
    field_image_invalid = auto()
    field_path_not_found = auto()
    field_path_not_group = auto()
    label_dataset_count_mismatch = auto()
    label_image_invalid = auto()
    label_image_source_invalid = auto()
    label_image_source_not_found = auto()
    label_multiscale_count_mismatch = auto()
    label_non_integer_dtype = auto()
    label_path_not_found = auto()
    label_path_not_group = auto()
    labels_metadata_invalid = auto()
    labels_not_group = auto()
    series_invalid_image = auto()
    series_path_not_found = auto()
    series_path_not_group = auto()
    well_invalid = auto()
    well_path_not_found = auto()
    well_path_not_group = auto()

    def __str__(self) -> str:
        return self.name


@dataclass(slots=True)
class ValidationResult:
    """Result of a validation operation containing any errors found."""

    errors: list[ErrorDetails] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge this result with another, combining errors."""
        return ValidationResult(errors=self.errors + other.errors)

    def add_error(
        self,
        error_type: StorageErrorType,
        loc: tuple[int | str, ...],
        msg: str,
        input_val: Any = None,
        ctx: dict[str, Any] | None = None,
        url: str | None = None,
    ) -> ValidationResult:
        """Add an error to this result and return self for chaining."""
        error: ErrorDetails = {
            "type": str(error_type),
            "loc": loc,
            "msg": msg,
            "input": input_val,
        }
        if ctx is not None:
            error["ctx"] = ctx
        if url is not None:
            error["url"] = url
        self.errors.append(error)
        return self

    @property
    def is_valid(self) -> bool:
        """Return True if no errors were found."""
        return len(self.errors) == 0
