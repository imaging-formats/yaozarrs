from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from typing_extensions import NotRequired, TypedDict

from yaozarrs._validation_warning import ValidationWarning
from yaozarrs._zarr import ZarrGroup, open_group

if TYPE_CHECKING:
    from pathlib import Path


def validate_zarr_store(obj: ZarrGroup | str | Path | Any) -> ZarrGroup:
    """Validate both structure and metadata of an OME-Zarr hierarchy.

    !!!important
        Requires `yaozarrs[io]` or `fsspec` to be installed.

    This is a high-level function to validate both the metadata and the structure
    of a complete OME-Zarr store.  This is the function used by the `yaozarrs validate`
    CLI command.

    Currently only supports OME-Zarr version 0.5.

    Parameters
    ----------
    obj : OMEZarrGroupJSON | ZarrGroup | str | Path | Any
        The zarr store to validate. Can be a URI string, a Path, a parsed
        OMEZarrGroupJSON object, a ZarrGroup instance, or a zarr.Group object
        (for backwards compatibility).

    Returns
    -------
    ZarrGroup
        The opened ZarrGroup if validation is successful.

    Raises
    ------
    StorageValidationError
        If the storage structure is invalid.
    """
    zarr_group = open_group(obj)
    ome_version = zarr_group.ome_version()
    if not ome_version:
        raise ValueError(
            f"Unable to determine OME-Zarr version for {zarr_group}. "
            "Is this an OME-Zarr group?"
        )

    if ome_version == "0.5":
        from yaozarrs.v05._storage import StorageValidatorV05

        Validator = StorageValidatorV05
    elif ome_version == "0.4":
        from yaozarrs.v04._storage import StorageValidatorV04

        Validator = StorageValidatorV04

    else:
        raise NotImplementedError(
            f"Structural validation for OME-Zarr version {ome_version} is "
            "not implemented."
        )

    # Capture RuntimeWarnings from pydantic validators during validation
    # and convert them to StorageValidationWarning
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.filterwarnings("always", category=ValidationWarning)
        result = Validator.validate_group(zarr_group)

    # prepend captured warnings to the result
    for w in captured_warnings:
        details: ErrorDetails = {
            "type": "model_warning",
            "loc": (),
            "msg": str(w.message),
        }
        result.warnings.insert(0, details)

    # Emit warnings for SHOULD directives from structural validation
    if result.warnings:
        # Use a custom formatter to match exception style
        warning_instance = StorageValidationWarning(result.warnings)

        _original_formatwarning = warnings.formatwarning

        def _custom_formatwarning(message, category, *_, **__) -> str:
            # Format like an exception: module.Class: message
            return f"{category.__module__}.{category.__name__}: {message}\n"

        warnings.formatwarning = _custom_formatwarning  # type: ignore
        try:
            warnings.warn(warning_instance, stacklevel=2)
        finally:
            warnings.formatwarning = _original_formatwarning

    # Raise error if any validation issues found
    if not result.is_valid:
        raise StorageValidationError(result.errors)

    return zarr_group


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


class _ValidationMessageMixin:
    """Mixin for formatting validation errors and warnings."""

    _details: list[ErrorDetails]

    def _format_message(self) -> str:
        """Generate a readable message from all validation details.

        Format matches Pydantic's ValidationError style with storage-specific context:
        - First line: count and title
        - Each item: location (dot-notation) on one line, message with context on next
        """
        if not self._details:  # pragma: no cover
            return f"No validation {self._details_noun}"

        count = len(self._details)
        lines = [f"{count} validation {self._details_noun} for {self.title}"]

        for detail in self._details:
            # Format location as dot-separated path (e.g., "ome.plate.wells.0.path")
            loc_str = ".".join(str(x) for x in detail["loc"])
            lines.append(loc_str)

            # Build the context bracket content
            ctx_parts = [f"type={detail['type']}"]

            # Add context fields if present
            if ctx := detail.get("ctx", {}):
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
            msg = detail["msg"]
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
        """The title used in the heading of the formatted message."""
        return type(self).__qualname__

    @property
    def _details_noun(self) -> str:
        """The noun to use for the details (e.g., 'error(s)' or 'warning(s)')."""
        raise NotImplementedError

    def get_details(
        self,
        *,
        include_context: bool = True,
    ) -> list[ErrorDetails]:
        """
        Details about each validation issue.

        Parameters
        ----------
        include_context: bool
            Whether to include the context of each item.

        Returns
        -------
            A list of `ErrorDetails` for each validation issue.
        """
        filtered_details: list[ErrorDetails] = []
        for detail in self._details:
            filtered_detail: ErrorDetails = {
                "type": detail["type"],
                "loc": detail["loc"],
                "msg": detail["msg"],
            }
            if include_context and "ctx" in detail:
                filtered_detail["ctx"] = detail["ctx"]
            filtered_details.append(filtered_detail)
        return filtered_details


class StorageValidationError(_ValidationMessageMixin, ValueError):
    """`StorageValidationError` is raised when validation of zarr storage fails.

    It contains a list of errors which detail why validation failed.
    """

    def __init__(self, errors: list[ErrorDetails]) -> None:
        self._details = errors
        super().__init__(self._format_message())

    @property
    def _details_noun(self) -> str:
        return "error(s)"

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
        return self.get_details(include_context=include_context)


class StorageValidationWarning(_ValidationMessageMixin, UserWarning):
    """`StorageValidationWarning` is raised when validation finds SHOULD violations.

    It contains a list of warnings which detail recommendations that were not followed.
    """

    def __init__(self, warnings_list: list[ErrorDetails] | str) -> None:
        if isinstance(warnings_list, str):
            # Simple string message (e.g., from pydantic validator warnings)
            self._details = []
            super().__init__(warnings_list)
        else:
            self._details = warnings_list
            super().__init__(self._format_message())

    @property
    def _details_noun(self) -> str:
        return "warning(s)"

    def warnings(
        self,
        *,
        include_context: bool = True,
    ) -> list[ErrorDetails]:
        """
        Details about each warning in the validation warning.

        Parameters
        ----------
        include_context: bool
            Whether to include the context of each warning.

        Returns
        -------
            A list of `ErrorDetails` for each warning in the validation warning.
        """
        return self.get_details(include_context=include_context)


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
    """Result of a validation operation containing any errors and warnings found."""

    errors: list[ErrorDetails] = field(default_factory=list)
    warnings: list[ErrorDetails] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge this result with another, combining errors and warnings."""
        return ValidationResult(
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )

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

    def add_warning(
        self,
        error_type: StorageErrorType,
        loc: tuple[int | str, ...],
        msg: str,
        *,
        ctx: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Add a warning to this result and return self for chaining.

        Warnings are for SHOULD directives - recommendations that don't invalidate
        the store but indicate best practices that should be followed.

        Parameters
        ----------
        error_type : StorageErrorType
            The type of warning that occurred.
        loc : tuple[int | str, ...]
            Location tuple identifying where in the metadata the warning occurred.
        msg : str
            Human-readable warning message.
        ctx : dict[str, Any] | None
            Additional context about the warning. Common fields:
            - fs_path: Filesystem path where the warning occurred
            - expected: What was expected
            - found/actual: What was actually found
        """
        warning: ErrorDetails = {"type": str(error_type), "loc": loc, "msg": msg}
        if ctx is not None:
            warning["ctx"] = ctx
        self.warnings.append(warning)
        return self

    @property
    def is_valid(self) -> bool:
        """Return True if no errors were found (warnings don't affect validity)."""
        return len(self.errors) == 0
