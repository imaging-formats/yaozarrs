from __future__ import annotations

import pytest
from pydantic import ValidationError

from yaozarrs import v05

WINDOW = {"start": 0, "end": 255, "min": 0, "max": 255}


@pytest.mark.parametrize(
    "color, expected",
    [
        # valid values are kept as-is (no case normalization)
        ("FF0000", "FF0000"),
        ("ff0000", "ff0000"),
        ("aBc123", "aBc123"),
        # obvious variants are repaired
        ("#ff0000", "ff0000"),
        ("#00FF00", "00FF00"),
        ("f00", "ff0000"),
        ("#0F0", "00FF00"),
    ],
)
def test_omero_color(color: str, expected: str) -> None:
    channel = v05.OmeroChannel(color=color, window=WINDOW)
    assert channel.color == expected


@pytest.mark.parametrize("color", ["red", "#12345", "ff00000", "gg0000", ""])
def test_invalid_omero_color(color: str) -> None:
    with pytest.raises(ValidationError, match=f"Invalid hex color code: {color!r}"):
        v05.OmeroChannel(color=color, window=WINDOW)
