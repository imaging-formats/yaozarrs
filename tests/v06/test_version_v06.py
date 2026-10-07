import pytest
from pydantic import TypeAdapter, ValidationError

from yaozarrs.v06._version import OMEV06

TA = TypeAdapter(OMEV06)


@pytest.mark.parametrize("v", ["0.6", "0.6.0", "0.6.dev4", "0.6.dev5", "0.6.dev10"])
def test_accepts_06_line(v: str) -> None:
    assert TA.validate_python(v) == v


@pytest.mark.parametrize("v", ["0.6.1", "0.6.2", "0.6.10", "0.5", "0.66", "0.7", "1.0"])
def test_rejects_non_06(v: str) -> None:
    with pytest.raises(ValidationError):
        TA.validate_python(v)
