"""Guard against unintended drift in the models of released spec versions.

v0.4 and v0.5 are released specs: their models should only change through
deliberate bug fixes.  Since some building blocks are shared across versions
(see `yaozarrs._common`), this test snapshots the JSON schema of every public
model in each frozen version and fails if any field, type, alias, default, or
constraint changes.  (Descriptions/titles are stripped, so docstring edits are ok.)

This does NOT capture custom validator logic; that is covered by the
per-version test suites (tests/v04, tests/v05).

If a change is intentional (e.g. a `fix(v05): ...` commit), regenerate with:

    YAOZARRS_UPDATE_FROZEN=1 uv run pytest tests/test_frozen_schemas.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import VERSION, TypeAdapter

from yaozarrs import v04, v05

if TYPE_CHECKING:
    from types import ModuleType

    from pydantic.json_schema import JsonSchemaMode

FROZEN_DIR = Path(__file__).parent / "frozen"
PYDANTIC_MINOR = ".".join(VERSION.split(".")[:2])
# json-schema keywords that only carry documentation
_DOC_KEYS = {"description", "title", "examples"}
# json-schema keywords whose values map *names* (not keywords) to subschemas
_NAME_MAPS = {"properties", "$defs", "patternProperties"}


def _strip_docs(obj: Any, names: bool = False) -> Any:
    if isinstance(obj, dict):
        if names:
            return {k: _strip_docs(v) for k, v in obj.items()}
        return {
            k: _strip_docs(v, names=k in _NAME_MAPS)
            for k, v in obj.items()
            if k not in _DOC_KEYS
        }
    if isinstance(obj, list):
        return [_strip_docs(v) for v in obj]
    return obj


def _public_schemas(mod: ModuleType) -> dict[str, Any]:
    """Schemas of everything in `mod.__all__`, with shared `$defs`."""
    mode: JsonSchemaMode = "validation"
    items = [(n, mode, TypeAdapter(getattr(mod, n))) for n in mod.__all__]
    key_map, defs = TypeAdapter.json_schemas(items)
    public = {key[0]: schema for key, schema in key_map.items()}
    return {"public": _strip_docs(public, names=True), **_strip_docs(defs)}


@pytest.mark.parametrize("mod", [v04, v05], ids=lambda m: m.__name__)
def test_frozen_schemas(mod: ModuleType) -> None:
    path = FROZEN_DIR / f"{mod.__name__.rsplit('.', 1)[-1]}.json"
    current = _public_schemas(mod)

    if os.getenv("YAOZARRS_UPDATE_FROZEN"):
        FROZEN_DIR.mkdir(exist_ok=True)
        data = {"pydantic": PYDANTIC_MINOR, **current}
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return

    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.pop("pydantic") != PYDANTIC_MINOR and current != frozen:
        # pydantic's schema generation changes across releases; only compare
        # strictly when using the same pydantic minor version as the snapshot.
        pytest.skip(f"snapshot generated with a different pydantic than {VERSION}")

    assert list(current["public"]) == list(frozen["public"]), "public API changed"
    for name, schema in current["$defs"].items():
        assert schema == frozen["$defs"].get(name), (
            f"{name!r} in {mod.__name__} drifted from its frozen schema. "
            "If intentional, see the docstring of this test module."
        )
    assert current == frozen
