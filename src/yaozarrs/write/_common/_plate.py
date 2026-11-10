"""Plate metadata auto-generation from `(row, column, field)` keys."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from ._models import check_model

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

    index = 0
    for char in row_name:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _row_index_to_name(index: int) -> str:
    """Convert row index to name, the inverse of `_row_name_to_index` (up to "ZZ")."""
    if index < 26:
        return chr(ord("A") + index)
    return chr(ord("A") + index // 26 - 1) + chr(ord("A") + index % 26)


def _column_name_to_index(col_name: str) -> int:
    """Convert column name to index (1=0, 2=1, 10=9, etc.)."""
    try:
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


def autogenerate_plate_metadata(
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
    fov_paths : Iterable[tuple[str, str, str]]
        `(row, col, fov)` keys (e.g. those of an images mapping).

    Returns
    -------
    dict[str, Any]
        Dictionary with 'rows', 'columns', 'wells' keys for PlateDef.
    """
    row_names: set[str] = set()
    col_names: set[str] = set()
    wells_set: set[tuple[str, str]] = set()
    for row, col, _fov in fov_paths:
        row_names.add(row)
        col_names.add(col)
        wells_set.add((row, col))

    max_row_idx = max(_row_name_to_index(r) for r in row_names)
    max_col_idx = max(_column_name_to_index(c) for c in col_names)
    width = _column_name_width(col_names)

    rows = [{"name": _row_index_to_name(i)} for i in range(max_row_idx + 1)]
    columns = [{"name": str(i + 1).zfill(width)} for i in range(max_col_idx + 1)]
    # (only the wells that have images)
    wells = [
        {
            "path": f"{row}/{col}",
            "rowIndex": _row_name_to_index(row),
            "columnIndex": _column_name_to_index(col),
        }
        for row, col in sorted(wells_set)
    ]

    return {"rows": rows, "columns": columns, "wells": wells}


def merge_plate_metadata(
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
    # (generated even for a given Plate: this also checks the images' row and
    # column names)
    auto_metadata = autogenerate_plate_metadata(images)

    if isinstance(user_plate, plate_type):
        _validate_plate_matches_images(user_plate, images)
        return user_plate
    if user_plate is not None and not isinstance(user_plate, Mapping):
        check_model(user_plate, plate_type, "plate: ")  # (raises)

    merged = {**auto_metadata, **cast("Mapping[str, Any]", user_plate or {})}
    plate = plate_type.model_validate({"plate": merged})
    _validate_plate_matches_images(plate, images)
    return plate


def extend_plate_metadata(plate: Any, new: Mapping[str, Any]) -> Any:
    """Add the rows, columns and wells of `new` that `plate` doesn't have yet."""
    merged = plate.plate.model_dump(mode="json", exclude_none=True)
    for key in ("rows", "columns"):
        have = {item["name"] for item in merged[key]}
        merged[key] += [item for item in new[key] if item["name"] not in have]
    rows = [row["name"] for row in merged["rows"]]
    cols = [col["name"] for col in merged["columns"]]
    for well in new["wells"]:
        row, col = well["path"].split("/")
        merged["wells"].append(
            {
                "path": well["path"],
                "rowIndex": rows.index(row),
                "columnIndex": cols.index(col),
            }
        )
    data = plate.model_dump(mode="json", exclude_none=True)
    return type(plate).model_validate({**data, "plate": merged})


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
    valid_rows = {row.name for row in plate.plate.rows}
    valid_cols = {col.name for col in plate.plate.columns}
    valid_wells = {well.path for well in plate.plate.wells}

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
