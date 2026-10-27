from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import VERSION, BaseModel, ConfigDict, Field

__all__ = ["_BaseModel"]

# populate_by_name is deprecated in pydantic v2.11+ (removed in v3).
# For >=2.11 use validate_by_name + validate_by_alias instead.
_PYDANTIC_V2_11 = tuple(int(x) for x in VERSION.split(".")[:2]) >= (2, 11)
_by_name_cfg: dict = (
    {"validate_by_name": True, "validate_by_alias": True}
    if _PYDANTIC_V2_11
    else {"populate_by_name": True}
)


class _BaseModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        validate_default=True,
        serialize_by_alias=True,
        **_by_name_cfg,
    )

    if not TYPE_CHECKING:
        # "by_alias" is required for round-tripping on pydantic <2.10.0
        #
        # "exclude_none": we checked that `null` never carries meaning for any field
        # defined by the NGFF spec (v0.4 - v0.6): none of the JSON schemas allow
        # `null` anywhere, and where the prose mentions it (`path` "MAY be omitted or
        # null") it is equivalent to omitting the key.  Optional fields here default
        # to `None` meaning "unset", so dumping them would write `null`s that the
        # official schemas reject.
        # Caveat: this also drops *user-defined* extra keys whose value is `null`
        # (e.g. a label property `{"label-value": 1, "class": null}`); pass
        # `exclude_none=False` to keep them.
        def model_dump_json(self, **kwargs: Any) -> str:
            kwargs.setdefault("by_alias", True)
            kwargs.setdefault("exclude_none", True)
            return super().model_dump_json(**kwargs)

        def model_dump(self, **kwargs: Any) -> str:  # pragma: no cover
            kwargs.setdefault("by_alias", True)
            kwargs.setdefault("exclude_none", True)
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
        exclude=True,  # don't include in serialization
    )
