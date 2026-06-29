"""Validate yaozarrs.v06 models against the real OME-NGFF spec examples.

The fixtures in ``tests/data/v06/examples`` are taken verbatim (comments
stripped) from ``ome/ngff-spec/examples`` on the ``main`` branch (the v0.6
source of truth). This is the strongest correctness check for the v06 models.
"""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from yaozarrs import v06, validate_ome_json

EXAMPLES = Path(__file__).parent.parent / "data" / "v06" / "examples"

# top-level zarr.json documents -> validate as OMEZarrGroupJSON
DOCUMENT_FILES = sorted(
    p
    for p in EXAMPLES.glob("*.json")
    if p.name != "multiscales_strict_multiscales_reference_to_label.json"
)
TRANSFORM_FILES = sorted((EXAMPLES / "transformations").glob("*.json"))


@pytest.mark.parametrize("path", DOCUMENT_FILES, ids=lambda p: p.name)
def test_spec_example_documents(path: Path) -> None:
    obj = validate_ome_json(path.read_text(), v06.OMEZarrGroupJSON)
    assert obj.attributes.ome is not None


def test_reference_to_label_fragment() -> None:
    # this example is stored as a bare {"ome": {...}} fragment
    data = json.loads(
        (
            EXAMPLES / "multiscales_strict_multiscales_reference_to_label.json"
        ).read_text()
    )
    obj = v06.OMEAttributes.model_validate(data)
    assert isinstance(obj.ome, v06.Image)


@pytest.mark.parametrize("path", TRANSFORM_FILES, ids=lambda p: p.name)
def test_spec_example_transformations(path: Path) -> None:
    data = json.loads(path.read_text())
    if "coordinateSystems" in data:
        TypeAdapter(list[v06.CoordinateSystem]).validate_python(
            data["coordinateSystems"]
        )
    transforms = TypeAdapter(list[v06.Transformation]).validate_python(
        data["coordinateTransformations"]
    )
    assert len(transforms) >= 1
