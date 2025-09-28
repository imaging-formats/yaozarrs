from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["_BaseModel"]


class _BaseModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        validate_default=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )

    if not TYPE_CHECKING:
        # "by_alias" is required for round-tripping on pydantic <2.10.0
        def model_dump_json(self, **kwargs: Any) -> str:
            kwargs.setdefault("by_alias", True)
            return super().model_dump_json(**kwargs)

        def model_dump(self, **kwargs: Any) -> str:  # pragma: no cover
            kwargs.setdefault("by_alias", True)
            return super().model_dump(**kwargs)


class ZarrGroupModel(_BaseModel):
    """Base class for models that have a direct mapping to a file or URI.

    e.g. v04 .zattrs or v05 zarr.json

    See Also
    --------
    v04.ZarrGroupJSON
    v05.ZarrGroupJSON
    """

    uri: str | None = Field(
        default=None,
        description=(
            "The URI this model was loaded from, if any. Note, if `from_uri()` is "
            "used, and a group directory is given, uri will resolve to the actual "
            "JSON file inside that directory that corresponds to this model."
        ),
        examples=[
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr/zarr.json",
            "/path/to/some_file.zarr/zarr.json",
        ],
    )
