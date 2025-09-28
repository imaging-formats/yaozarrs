from typing import Literal

from pydantic import Field

from yaozarrs._base import ZarrGroupModel


class Bf2Raw(ZarrGroupModel):
    bioformats2raw_layout: Literal[3] = Field(
        alias="bioformats2raw.layout",
        description="The top-level identifier metadata added by bioformats2raw",
    )
