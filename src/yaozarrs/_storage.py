from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
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
    """Tuple of str and ints identifying where in the metadata the error occurred."""
    msg: str
    """A human readable error message."""
    ctx: NotRequired[dict[str, Any]]
    """
    Additional context about the error, specific to the validation task.

    Common context fields for storage validation:
    - fs_path: Filesystem path in the zarr store where the error occurred
    - expected: What was expected (type, value, or state)
    - found/actual: What was actually found
    - missing: What required element is missing
    - *_ndim, *_count: Specific numeric mismatches
    - error: Exception object
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
        """Generate a readable error message from all validation errors.

        Format matches Pydantic's ValidationError style with storage-specific context:
        - First line: error count and title
        - Each error: location (dot-notation) on one line, message with context on next
        """
        if not self._errors:  # pragma: no cover
            return "No validation errors"

        lines = [f"{len(self._errors)} validation error(s) for {self.title}"]

        for error in self._errors:
            # Format location as dot-separated path (e.g., "ome.plate.wells.0.path")
            loc_str = ".".join(str(x) for x in error["loc"])
            lines.append(loc_str)

            # Build the context bracket content
            ctx_parts = [f"type={error['type']}"]

            # Add context fields if present
            if ctx := error.get("ctx", {}):
                # Add fs_path first if present (most important for debugging)
                if "fs_path" in ctx:
                    ctx_parts.append(f"fs_path={ctx['fs_path']!r}")

                # Add expected/found/actual/missing fields
                for key in ("expected", "found", "actual"):
                    if key in ctx:
                        val = ctx[key]
                        ctx_parts.append(f"{key}={val!r}")

                # Add any other context fields (like *_ndim, *_count, etc.)
                # Skip 'error' field to avoid duplication in display
                for key, val in ctx.items():
                    if key not in (
                        "fs_path",
                        "expected",
                        "found",
                        "actual",
                        "error",
                    ):
                        ctx_parts.append(f"{key}={val!r}")

            ctx_str = ", ".join(ctx_parts)

            # Message line with 2-space indent
            # Use textwrap.indent to handle multi-line messages
            msg = error["msg"]
            msg_with_context = f"{msg} [{ctx_str}]"
            indented_msg = textwrap.indent(msg_with_context, "  ")
            if "error" in ctx:
                ctx_error = ctx["error"]
                if isinstance(ctx_error, ValidationError):
                    # If the error is a Pydantic ValidationError, include its errors
                    nested_errors = textwrap.indent(str(ctx_error), "  ")
                    indented_msg += f"\n{nested_errors}"

            lines.append(indented_msg)
            lines.append("")

        return "\n".join(lines)

    @property
    def title(self) -> str:
        """The title of the error, as used in the heading of `str(validation_error)`."""
        return "StorageValidationError"

    def errors(
        self,
        *,
        include_context: bool = True,
    ) -> list[ErrorDetails]:
        """
        Details about each error in the validation error.

        Parameters
        ----------
        include_context: bool
            Whether to include the context of each error.

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
            if include_context and "ctx" in error:
                filtered_error["ctx"] = error["ctx"]
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
        *,
        ctx: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Add an error to this result and return self for chaining.

        Parameters
        ----------
        error_type : StorageErrorType
            The type of error that occurred.
        loc : tuple[int | str, ...]
            Location tuple identifying where in the metadata the error occurred.
        msg : str
            Human-readable error message.
        ctx : dict[str, Any] | None
            Additional context about the error. Common fields:
            - fs_path: Filesystem path where the error occurred
            - expected: What was expected
            - found/actual: What was actually found
            - missing: What required element is missing
        """
        error: ErrorDetails = {"type": str(error_type), "loc": loc, "msg": msg}
        if ctx is not None:
            error["ctx"] = ctx
        self.errors.append(error)
        return self

    @property
    def is_valid(self) -> bool:
        """Return True if no errors were found."""
        return len(self.errors) == 0
