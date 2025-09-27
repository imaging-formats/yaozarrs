from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

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

        def model_dump_json(self, **kwargs: Any) -> str:
            # but required for round-tripping on pydantic <2.10.0
            kwargs.setdefault("by_alias", True)
            return super().model_dump_json(**kwargs)

        def model_dump(self, **kwargs: Any) -> str:  # pragma: no-cover
            # but required for round-tripping on pydantic <2.10.0
            kwargs.setdefault("by_alias", True)
            return super().model_dump(**kwargs)


class ZarrGroupModel(_BaseModel):
    """Base class for models that have a direct mapping to a json file.

    e.g. v04 .zattrs or v05 zarr.json

    See Also
    --------
    v04.ZarrGroupJSON
    v05.ZarrGroupJSON
    """

    uri: str | None = Field(
        default=None,
        description="The URI this model was loaded from, if any.",
        examples=[
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr",
            "/path/to/some_file.zarr",
        ],
    )

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Create an instance of this model by loading JSON data from a URI.

        Parameters
        ----------
        uri : str
            The URI to load the JSON data from.  This can be a local file path,
            or a remote URL (e.g. s3://bucket/key/some_file.zarr).

        Returns
        -------
        Self
            An instance of this model loaded from the URI.

        Raises
        ------
        FileNotFoundError
            If the URI or required zarr.json file cannot be found.
        """
        try:
            import fsspec
        except ImportError as e:
            msg = (
                "fsspec is required for from_uri. "
                "Install with: pip install yaozarrs[io]"
            )
            raise ImportError(msg) from e

        # Determine the target JSON file URI
        if uri.endswith((".json", ".zattrs")):
            json_uri = uri
        else:
            # Assume it's a directory, look for zarr.json
            json_uri = f"{uri.rstrip('/')}/zarr.json"

        # Load JSON content using fsspec
        try:
            with fsspec.open(json_uri, "r") as f:
                json_content = f.read()
        except Exception as e:
            msg = f"Could not load JSON from URI: {json_uri}"
            raise FileNotFoundError(msg) from e

        # Create instance and set the original URI
        instance = cls.model_validate_json(json_content)
        instance.uri = uri
        return instance
