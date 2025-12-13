import importlib.util
from pathlib import Path

import pytest

from yaozarrs import v04, v05, validate_ome_uri
from yaozarrs._base import ZarrGroupModel

try:
    from aiohttp.client_exceptions import ClientConnectorError

    connection_exceptions: tuple[type[Exception], ...] = (ClientConnectorError,)
except ImportError:
    connection_exceptions = ()

HAVE_FSSPEC = importlib.util.find_spec("fsspec")
DATA = Path(__file__).parent / "data"
V05_DATA = DATA / "v05"

SOURCES = {
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr": v05.Image,  # noqa E501
    V05_DATA / "6001240_labels.zarr": v05.Image,
    "https://uk1s3.embassy.ebi.ac.uk/idr/share/ome2024-ngff-challenge/idr0010/76-45.zarr": v05.Plate,  # noqa E501
    V05_DATA / "76-45.ome.zarr": v05.Plate,
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr": v04.Image,
}


@pytest.mark.skipif(not HAVE_FSSPEC, reason="fsspec not installed")
@pytest.mark.parametrize("uri,expected_type", SOURCES.items())
def test_from_uri(uri: str, expected_type: type) -> None:
    try:
        obj = validate_ome_uri(uri)
    except connection_exceptions as e:
        pytest.xfail(reason=f"Internet down?: {e}")
        return

    if isinstance(obj, v05.OMEZarrGroupJSON):
        # in v05, the ome info is inside the doc['attributes']['ome'] key
        assert isinstance(obj.attributes.ome, expected_type)
        assert obj.uri is not None and obj.uri.endswith("zarr.json")
    else:
        # in v04, the document is itself the ome model
        assert isinstance(obj, expected_type)
        assert isinstance(obj, ZarrGroupModel)
        assert obj.uri is not None and obj.uri.endswith(".zattrs")
