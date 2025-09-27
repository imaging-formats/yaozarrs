from pathlib import Path

import pytest

from yaozarrs.v05 import OMEZarrGroupJSON

DATA = Path(__file__).parent / "data"
# ALL of the zarr.json files in the test data
ZARR_JSONS = sorted(x for x in DATA.rglob("zarr.json"))
# The *contents* of all zarr.json files that contain OME metadata
OME_ZARR_JSONS: dict[str, str] = {
    str(path.relative_to(DATA)): content
    for path in ZARR_JSONS
    if '"ome"' in (content := path.read_text())
}


@pytest.mark.parametrize("txt", OME_ZARR_JSONS.values(), ids=OME_ZARR_JSONS.keys())
def test_data(txt: str) -> None:
    obj = OMEZarrGroupJSON.model_validate_json(txt)
    js = obj.model_dump_json()
    obj2 = OMEZarrGroupJSON.model_validate_json(js)
    assert obj == obj2
