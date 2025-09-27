from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

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
