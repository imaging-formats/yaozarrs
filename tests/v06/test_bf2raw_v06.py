from __future__ import annotations

import pytest
from pydantic import ValidationError

from yaozarrs import v06


def test_bf2raw_and_series() -> None:
    b = v06.Bf2Raw.model_validate({"version": "0.6", "bioformats2raw.layout": 3})
    assert b.model_dump(mode="json") == {"version": "0.6", "bioformats2raw.layout": 3}
    s = v06.Series.model_validate({"version": "0.6", "series": ["0", "1"]})
    assert s.series == ["0", "1"]
    with pytest.raises(ValidationError, match="at least 1 item"):
        v06.Series.model_validate({"version": "0.6", "series": []})
