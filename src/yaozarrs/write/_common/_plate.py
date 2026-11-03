"""Plate metadata auto-generation from `(row, column, field)` keys."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from ._models import _check_model

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel

    from ._builders import ImageWithAny


def _row_name_to_index(row_name: str) -> int:
    """Convert row name to index (A=0, B=1, ..., Z=25, AA=26, etc.)."""
    if not row_name or not row_name.isalpha() or not row_name.isupper():
        raise ValueError(  # pragma: no cover
            f"Row name must be uppercase letters (A-Z, AA-ZZ, etc.), got: {row_name}"
        )

    # Convert like Excel columns: A=0, B=1, ..., Z=25, AA=26, AB=27, etc.
    index = 0
    for char in row_name:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _column_name_to_index(col_name: str) -> int:
    """Convert column name to index (1=0, 2=1, 10=9, etc.)."""
    try:
        # Try parsing as integer (handles "1", "2", "10", "1", etc.)
        return int(col_name) - 1
    except ValueError:  # pragma: no cover
        raise ValueError(
            f"Column name must be numeric (1, 2, 10, etc.), got: {col_name}"
        ) from None


def _column_name_width(col_names: Iterable[str]) -> int:
    """Return the zero-padded width of numeric column names (0 if unpadded)."""
    names = sorted(col_names)
    padded = [c for c in names if len(c) > 1 and c.startswith("0")]
    if not padded:
        return 0
    if len({len(c) for c in names}) > 1:
        raise ValueError(
            "Column names mix zero-padded and other widths "
            f"({', '.join(map(repr, names))}), so the names of the columns in "
            "between can't be inferred.  Use the same width for all of them, or "
            "pass `plate=` with explicit columns."
        )
    return len(padded[0])


def _autogenerate_plate_metadata(
    fov_paths: Iterable[tuple[str, str, str]],
) -> dict[str, Any]:
    """Auto-generate plate metadata from images dict keys.

    Row indices follow the convention: A=0, B=1, ..., Z=25, AA=26, etc.
    Column indices follow: 1=0, 2=1, ..., 10=9, etc.  Zero-padded column names
    (e.g. "01", "02") are kept, and the generated columns use the same width.

    All intermediate rows/columns are created (e.g., if you have rows A and D,
    rows B and C are also created, even if they have no wells).

    Parameters
    ----------
    fov_paths : Mapping[tuple[str, str, str], ImageWithAny]
        Images mapping with (row, col, fov) keys.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'rows', 'columns', 'wells' keys for PlateDef.
    """
    # Extract unique row and column names
    rows_set: set[str] = set()
    col_names: set[str] = set()
    wells_set: set[tuple[str, str]] = set()
    for row, col, _fov in fov_paths:
        rows_set.add(row)
        col_names.add(col)
        wells_set.add((row, col))

    # Find the maximum row and column to fill in all intermediates
    max_row_idx = max(_row_name_to_index(r) for r in rows_set)
    max_col_idx = max(_column_name_to_index(c) for c in col_names)
    width = _column_name_width(col_names)

    # Create all rows from A to max_row (e.g., A, B, C, D if max is D)
    rows = []
    for idx in range(max_row_idx + 1):
        # Convert index back to row name (0=A, 1=B, etc.)
        if idx < 26:
            row_name = chr(ord("A") + idx)
        else:
            # Handle AA, AB, etc. (Excel-style)
            first = chr(ord("A") + (idx // 26) - 1)
            second = chr(ord("A") + (idx % 26))
            row_name = first + second
        rows.append({"name": row_name})

    # Create all columns from 1 to max_col (e.g., 1, 2, 3, 4, 5 if max is 5),
    # zero-padded like the given names (e.g. 01, 02, ... 12)
    columns = [{"name": str(i + 1).zfill(width)} for i in range(max_col_idx + 1)]

    # Create well objects only for wells that have images
    wells = [
        {
            "path": f"{row}/{col}",
            "rowIndex": _row_name_to_index(row),
            "columnIndex": _column_name_to_index(col),
        }
        for row, col in sorted(wells_set)
    ]

    return {"rows": rows, "columns": columns, "wells": wells}


def _merge_plate_metadata(
    images: Mapping[tuple[str, str, str], ImageWithAny],
    user_plate: BaseModel | Mapping[str, Any] | None,
    plate_type: type[Any],
) -> Any:
    """Merge auto-generated and user-provided plate metadata.

    Parameters
    ----------
    images : Mapping[tuple[str, str, str], ImageWithAny]
        Images mapping used to auto-generate metadata.
    user_plate : Plate | dict[str, Any] | None
        User-provided plate metadata (takes precedence over auto-generated).
    plate_type : type[Plate]
        The (version-specific) Plate model to construct.

    Returns
    -------
    Plate
        Final Plate object with merged metadata.

    Raises
    ------
    ValueError
        If user-provided metadata conflicts with images dict.
    """
    # Auto-generate base metadata from images
    auto_metadata = _autogenerate_plate_metadata(images)

    # If user provided a full Plate object, validate and return
    if isinstance(user_plate, plate_type):
        _validate_plate_matches_images(user_plate, images)
        return user_plate
    if user_plate is not None and not isinstance(user_plate, Mapping):
        _check_model(user_plate, plate_type, "plate: ")  # (raises)

    # If user provided a dict, merge with auto-generated
    merged = {**auto_metadata, **cast("Mapping[str, Any]", user_plate or {})}

    # Construct Plate object from merged metadata
    plate = plate_type.model_validate({"plate": merged})

    # Validate that all images have valid coordinates
    _validate_plate_matches_images(plate, images)

    return plate


def _validate_plate_matches_images(
    plate: Any,
    images: Mapping[tuple[str, str, str], ImageWithAny],
) -> None:
    """Validate that plate metadata matches images dict.

    Parameters
    ----------
    plate : Plate
        Plate metadata to validate.
    images : Mapping[tuple[str, str, str], ImageWithAny]
        Images mapping to validate against.

    Raises
    ------
    ValueError
        If any image coordinates don't match plate metadata.
    """
    # Get valid row and column names from plate
    valid_rows = {row.name for row in plate.plate.rows}
    valid_cols = {col.name for col in plate.plate.columns}
    valid_wells = {well.path for well in plate.plate.wells}

    # Check all images have valid coordinates
    for row, col, _fov in images.keys():
        if row not in valid_rows:  # pragma: no cover
            raise ValueError(
                f"Image row '{row}' not found in plate rows: {sorted(valid_rows)}"
            )
        if col not in valid_cols:  # pragma: no cover
            raise ValueError(
                f"Image column '{col}' not found in plate columns: {sorted(valid_cols)}"
            )
        well_path = f"{row}/{col}"
        if well_path not in valid_wells:  # pragma: no cover
            raise ValueError(
                f"Image well '{well_path}' not found in plate wells: "
                f"{sorted(valid_wells)}"
            )
