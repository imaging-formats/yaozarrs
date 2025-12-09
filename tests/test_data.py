from pathlib import Path

import pytest

from yaozarrs import validate_ome_json

DATA = Path(__file__).parent / "data"

# ALL of the zarr.json files in the test data
ZARR_JSONS = sorted(x for x in DATA.rglob("zarr.json") if "broken" not in str(x))
# The *contents* of all zarr.json files that contain OME metadata
OME_ZARR_JSONS: dict[str, str] = {
    str(path.relative_to(DATA)): content
    for path in ZARR_JSONS
    if '"ome"' in (content := path.read_text())
}

ZATTRS = sorted(x for x in DATA.rglob(".zattrs"))
# in v04.  the .zattrs file ITSELF was the ome document.
# there's no quick way to filter here based on the presence of an "ome" key
OME_ZARR_ZATTRS: dict[str, str] = {
    str(path.relative_to(DATA)): path.read_text() for path in ZATTRS
}

PATHs, TXTs = zip(*{**OME_ZARR_JSONS, **OME_ZARR_ZATTRS}.items())


@pytest.mark.parametrize("txt", TXTs, ids=PATHs)
def test_data(txt: str) -> None:
    obj = validate_ome_json(txt)
    js = obj.model_dump_json()
    obj2 = validate_ome_json(js)
    assert obj == obj2
