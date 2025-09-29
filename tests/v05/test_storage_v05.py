from yaozarrs.v05._storage import validate_zarr_store


def test_validate_storage() -> None:
    validate_zarr_store("~/Downloads/6001240_labels.zarr")
