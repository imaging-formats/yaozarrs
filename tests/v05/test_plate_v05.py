import pytest
from pydantic import ValidationError

from yaozarrs import v05, validate_ome_object

# Helper data
COLUMN_A = {"name": "01"}
COLUMN_B = {"name": "02"}
ROW_A = {"name": "A"}
ROW_B = {"name": "B"}

WELL_A01 = {"path": "A/01", "rowIndex": 0, "columnIndex": 0}
WELL_A02 = {"path": "A/02", "rowIndex": 0, "columnIndex": 1}
WELL_B01 = {"path": "B/01", "rowIndex": 1, "columnIndex": 0}
WELL_B02 = {"path": "B/02", "rowIndex": 1, "columnIndex": 1}

ACQUISITION_1 = {
    "id": 0,
    "name": "Acquisition 1",
    "description": "First acquisition",
    "maximumfieldcount": 1,
    "starttime": 1234567890,
    "endtime": 1234567950,
}

ACQUISITION_2 = {
    "id": 1,
    "name": "Acquisition 2",
    "maximumfieldcount": 2,
}

V05_VALID_PLATES = [
    # Minimal valid plate
    {
        "version": "0.5",
        "plate": {
            "columns": [COLUMN_A, COLUMN_B],
            "rows": [ROW_A, ROW_B],
            "wells": [WELL_A01, WELL_A02, WELL_B01, WELL_B02],
        },
    },
    # Plate with optional fields
    {
        "version": "0.5",
        "plate": {
            "name": "Test Plate",
            "field_count": 4,
            "columns": [COLUMN_A, COLUMN_B],
            "rows": [ROW_A, ROW_B],
            "wells": [WELL_A01, WELL_A02, WELL_B01, WELL_B02],
            "acquisitions": [ACQUISITION_1, ACQUISITION_2],
        },
    },
    # Plate with single column/row
    {
        "version": "0.5",
        "plate": {
            "columns": [COLUMN_A],
            "rows": [ROW_A],
            "wells": [{"path": "A/01", "rowIndex": 0, "columnIndex": 0}],
        },
    },
    # Plate with alphanumeric names
    {
        "version": "0.5",
        "plate": {
            "columns": [{"name": "Col1"}, {"name": "Col2"}],
            "rows": [{"name": "Row1"}, {"name": "Row2"}],
            "wells": [
                {"path": "Row1/Col1", "rowIndex": 0, "columnIndex": 0},
                {"path": "Row1/Col2", "rowIndex": 0, "columnIndex": 1},
                {"path": "Row2/Col1", "rowIndex": 1, "columnIndex": 0},
                {"path": "Row2/Col2", "rowIndex": 1, "columnIndex": 1},
            ],
        },
    },
]


@pytest.mark.parametrize("obj", V05_VALID_PLATES)
def test_valid_v05_plates(obj: dict) -> None:
    validate_ome_object(obj, v05.Plate)


# list of (invalid_obj, error_msg_substring)
V05_INVALID_PLATES: list[tuple[dict, str]] = [
    # Missing required fields
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                # missing wells
            },
        },
        "Field required",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                # missing rows
                "wells": [WELL_A01],
            },
        },
        "Field required",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                # missing columns
                "rows": [ROW_A],
                "wells": [WELL_A01],
            },
        },
        "Field required",
    ),
    # Empty required arrays
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [],  # empty
                "rows": [ROW_A],
                "wells": [WELL_A01],
            },
        },
        "at least 1 item",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [],  # empty
                "wells": [WELL_A01],
            },
        },
        "at least 1 item",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [],  # empty
            },
        },
        "at least 1 item",
    ),
    # Invalid column/row names (non-alphanumeric)
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [{"name": "Col-1"}],  # hyphen not allowed
                "rows": [ROW_A],
                "wells": [WELL_A01],
            },
        },
        "String should match pattern",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [{"name": "Row_A"}],  # underscore not allowed
                "wells": [WELL_A01],
            },
        },
        "String should match pattern",
    ),
    # Invalid well path patterns
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A-01",  # hyphen not allowed
                        "rowIndex": 0,
                        "columnIndex": 0,
                    }
                ],
            },
        },
        "String should match pattern",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A/01/extra",  # too many parts
                        "rowIndex": 0,
                        "columnIndex": 0,
                    }
                ],
            },
        },
        "String should match pattern",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A",  # missing column part
                        "rowIndex": 0,
                        "columnIndex": 0,
                    }
                ],
            },
        },
        "String should match pattern",
    ),
    # Well indices out of bounds
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A/01",
                        "rowIndex": 1,  # only 1 row (index 0)
                        "columnIndex": 0,
                    }
                ],
            },
        },
        "but only 1 rows exist",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A/01",
                        "rowIndex": 0,
                        "columnIndex": 1,  # only 1 column (index 0)
                    }
                ],
            },
        },
        "but only 1 columns exist",
    ),
    # Negative indices
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A/01",
                        "rowIndex": -1,
                        "columnIndex": 0,
                    }
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [
                    {
                        "path": "A/01",
                        "rowIndex": 0,
                        "columnIndex": -1,
                    }
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    # Invalid field_count (not positive)
    (
        {
            "version": "0.5",
            "plate": {
                "field_count": 0,
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01],
            },
        },
        "Input should be greater than 0",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "field_count": -1,
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01],
            },
        },
        "Input should be greater than 0",
    ),
    # Invalid acquisition fields
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01],
                "acquisitions": [
                    {
                        "id": -1,  # negative id
                        "name": "Test",
                    }
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01],
                "acquisitions": [
                    {
                        "id": 0,
                        "maximumfieldcount": 0,  # not positive
                    }
                ],
            },
        },
        "Input should be greater than 0",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01],
                "acquisitions": [
                    {
                        "id": 0,
                        "starttime": -1,  # negative timestamp
                    }
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01],
                "acquisitions": [
                    {
                        "id": 0,
                        "endtime": -1,  # negative timestamp
                    }
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    # Duplicate columns (UniqueList violation)
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A, COLUMN_A],  # duplicate
                "rows": [ROW_A],
                "wells": [WELL_A01],
            },
        },
        "List items are not unique",
    ),
    # Duplicate rows (UniqueList violation)
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A, ROW_A],  # duplicate
                "wells": [WELL_A01],
            },
        },
        "List items are not unique",
    ),
    # Duplicate wells (UniqueList violation)
    (
        {
            "version": "0.5",
            "plate": {
                "columns": [COLUMN_A],
                "rows": [ROW_A],
                "wells": [WELL_A01, WELL_A01],  # duplicate
            },
        },
        "List items are not unique",
    ),
]


@pytest.mark.parametrize("obj, msg", V05_INVALID_PLATES)
def test_invalid_v05_plates(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v05.Plate.model_validate(obj)
