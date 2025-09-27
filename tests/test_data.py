from pathlib import Path

import pytest

from yaozarrs.v05._zarr_json import ZarrJSON

DATA = Path(__file__).parent / "data"
# ALL of the zarr.json files in the test data
ZARR_JSONS = sorted(x for x in DATA.rglob("zarr.json"))
# The *contents* of all zarr.json files that contain OME metadata
OME_ZARR_JSONS: dict[str, str] = {
    str(path.relative_to(DATA)): content
    for path in ZARR_JSONS
    if '"ome"' in (content := path.read_text())
}


@pytest.mark.parametrize(
    "txt", OME_ZARR_JSONS.values(), ids=list(OME_ZARR_JSONS.keys())
)
def test_data(txt: str) -> None:
    obj = ZarrJSON.model_validate_json(txt)

    # FIXME:
    # we shouldn't have to do by_alias=True here...
    # but it's required for pydantic <2.10.0
    js = obj.model_dump_json(by_alias=True)
    obj2 = ZarrJSON.model_validate_json(js)
    assert obj == obj2
