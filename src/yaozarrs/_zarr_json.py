from typing import TypeAlias

from . import v04, v05

ZarrJSON: TypeAlias = v05.OMEZarrGroupJSON | v04.OMEZarrGroupJSON
