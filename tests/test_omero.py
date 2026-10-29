from __future__ import annotations

import pytest
from pydantic import ValidationError

from yaozarrs import v05
from yaozarrs._validation_warning import ValidationWarning

WINDOW = {"start": 0, "end": 255, "min": 0, "max": 255}


# valid values are kept as-is (no case normalization)
@pytest.mark.parametrize("color", ["FF0000", "ff0000", "aBc123"])
def test_omero_color(color: str) -> None:
    assert v05.OmeroChannel(color=color, window=WINDOW).color == color


# obvious variants are repaired, with a warning (they don't conform to the spec)
@pytest.mark.parametrize(
    "color, expected",
    [
        ("#ff0000", "ff0000"),
        ("#00FF00", "00FF00"),
        ("f00", "ff0000"),
        ("#0F0", "00FF00"),
    ],
)
def test_omero_color_repaired(color: str, expected: str) -> None:
    with pytest.warns(ValidationWarning, match="not 6 hexadecimal digits"):
        channel = v05.OmeroChannel(color=color, window=WINDOW)
    assert channel.color == expected


@pytest.mark.parametrize("color", ["red", "#12345", "ff00000", "gg0000", ""])
def test_invalid_omero_color(color: str) -> None:
    with pytest.raises(ValidationError, match=f"Invalid hex color code: {color!r}"):
        v05.OmeroChannel(color=color, window=WINDOW)
