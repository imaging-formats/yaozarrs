"""Convert OME-Zarr metadata between spec versions.

A conversion is a migration of the JSON document, and the target version's model is
the judge of the result: a model is dumped, migrated one adjacent version at a time,
and validated as the target model.  Version pairs without a step differ only in where
`version` lives, which the target model's defaults take care of.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias, overload

from . import v04, v05, v06

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["ConversionWarning", "convert_version"]

Version: TypeAlias = Literal["0.4", "0.5", "0.6"]

_PACKAGES = {"0.4": v04, "0.5": v05, "0.6": v06}
_ORDER = tuple(_PACKAGES)
_KINDS = (
    "Image",
    "LabelImage",
    "LabelsGroup",
    "Plate",
    "Well",
    "Bf2Raw",
    "Series",
    "Multiscale",
    "Scene",
)
# model class -> (the version it belongs to, its name in every version's package)
_MODELS: dict[type[BaseModel], tuple[str, str]] = {
    getattr(package, kind): (version, kind)
    for version, package in _PACKAGES.items()
    for kind in _KINDS
    if hasattr(package, kind)
}
_GROUPS = (v05.OMEZarrGroupJSON, v06.OMEZarrGroupJSON)
_ATTRIBUTES = (v05.OMEAttributes, v06.OMEAttributes)


class ConversionWarning(Warning):
    """Metadata was dropped because the target version can't express it."""


# ------------------------------------------------------------------------------
# Migration steps, on JSON-shaped documents
# ------------------------------------------------------------------------------


def _map_multiscales(doc: dict, func: Callable[[dict], dict]) -> dict:
    """Apply `func` to each multiscale of `doc`, which may itself be a multiscale."""
    if "datasets" in doc:
        return func(doc)
    if "multiscales" in doc:
        return {**doc, "multiscales": [func(ms) for ms in doc["multiscales"]]}
    return doc


def _upgrade_05_06(doc: dict, coordinate_system: str, dropped: list[str]) -> dict:
    return _map_multiscales(doc, lambda ms: _upgrade_multiscale(ms, coordinate_system))


def _upgrade_multiscale(ms: dict, coordinate_system: str) -> dict:
    ms = dict(ms)
    axes = ms.pop("axes")
    overall = _transform_params(ms.pop("coordinateTransformations", None) or [])
    ms["coordinateSystems"] = [{"name": coordinate_system, "axes": axes}]
    ms["datasets"] = [
        _upgrade_dataset(ds, overall, coordinate_system) for ds in ms["datasets"]
    ]
    return ms


def _upgrade_dataset(ds: dict, overall: dict, coordinate_system: str) -> dict:
    params = _transform_params(ds["coordinateTransformations"])
    scale, translation = params["scale"], params.get("translation")
    if overall:
        # multiscale transformations are "applied after" those of the datasets, and
        # the intrinsic coordinate system is the image's native physical one, so they
        # belong in the dataset's transformation.
        g_scale, g_translation = overall["scale"], overall.get("translation")
        if translation or g_translation:
            translation = [
                g * t + o
                for g, t, o in zip(
                    g_scale,
                    translation or [0.0] * len(scale),
                    g_translation or [0.0] * len(scale),
                )
            ]
        scale = [g * s for g, s in zip(g_scale, scale)]

    transform: dict[str, Any] = {"type": "scale", "scale": scale}
    if translation is not None:
        transform = {
            "type": "sequence",
            "transformations": [
                transform,
                {"type": "translation", "translation": translation},
            ],
        }
    transform["input"] = {"path": ds["path"]}
    transform["output"] = {"name": coordinate_system}
    return {**ds, "coordinateTransformations": [transform]}


def _transform_params(transforms: list[dict]) -> dict[str, list[float]]:
    """Return eg `{"scale": [...], "translation": [...]}` for a v0.5 transform list."""
    return {t["type"]: t[t["type"]] for t in transforms}


def _downgrade_06_05(doc: dict, coordinate_system: str, dropped: list[str]) -> dict:
    return _map_multiscales(doc, lambda ms: _downgrade_multiscale(ms, dropped))


def _downgrade_multiscale(ms: dict, dropped: list[str]) -> dict:
    ms = dict(ms)
    systems = ms.pop("coordinateSystems")
    intrinsic = ms["datasets"][0]["coordinateTransformations"][0]["output"]["name"]
    axes = next(cs for cs in systems if cs["name"] == intrinsic)["axes"]

    dropped.extend(
        f"coordinate system {cs['name']!r}" for cs in systems if cs["name"] != intrinsic
    )
    if ms.pop("coordinateTransformations", None):
        dropped.append("multiscale coordinateTransformations")
    dropped.extend(
        f"axis {ax['name']!r} {key}"
        for ax in axes
        for key in ("longName", "discrete")
        if ax.get(key) is not None
    )

    ms["axes"] = axes
    ms["datasets"] = [
        {**ds, "coordinateTransformations": _downgrade_transform(ds, len(axes))}
        for ds in ms["datasets"]
    ]
    return ms


def _downgrade_transform(ds: dict, ndim: int) -> list[dict]:
    transform = ds["coordinateTransformations"][0]
    if transform["type"] == "identity":
        return [{"type": "scale", "scale": [1.0] * ndim}]
    if transform["type"] == "sequence":
        return list(transform["transformations"])
    return [transform]


def _downgrade_05_04(doc: dict, coordinate_system: str, dropped: list[str]) -> dict:
    # (only v0.5 onwards allows additional keys under `colors`)
    colors = (doc.get("image-label") or {}).get("colors") or []
    known = {"label-value", "rgba"}
    extras = dict.fromkeys(key for color in colors for key in color.keys() - known)
    dropped.extend(f"label color key {key!r}" for key in extras)
    return doc


_STEPS = {
    ("0.5", "0.6"): _upgrade_05_06,
    ("0.6", "0.5"): _downgrade_06_05,
    ("0.5", "0.4"): _downgrade_05_04,
}


def _strip_versions(doc: dict) -> dict:
    """Remove the spec's `version` keys (the target's defaults replace them).

    Must not recurse: `multiscales[].metadata` commonly has a tool's own `version`.
    """
    doc = {k: v for k, v in doc.items() if k != "version"}
    for key in ("plate", "well", "image-label"):
        if isinstance(doc.get(key), dict):
            doc[key] = {k: v for k, v in doc[key].items() if k != "version"}
    if "multiscales" in doc:
        doc["multiscales"] = [
            {k: v for k, v in ms.items() if k != "version"} for ms in doc["multiscales"]
        ]
    return doc


def _convert(obj: Any, version: str, coordinate_system: str, dropped: list[str]) -> Any:
    package = _PACKAGES[version]
    if isinstance(obj, _GROUPS):
        attrs = _convert(obj.attributes, version, coordinate_system, dropped)
        if not hasattr(package, "OMEAttributes"):
            return attrs
        return package.OMEZarrGroupJSON(attributes=attrs)
    if isinstance(obj, _ATTRIBUTES):
        ome = _convert(obj.ome, version, coordinate_system, dropped)
        extra = copy.deepcopy(obj.model_extra or {})
        if not hasattr(package, "OMEAttributes"):
            dropped.extend(f"attribute {key!r}" for key in extra)
            return ome
        return package.OMEAttributes.model_validate({**extra, "ome": ome})

    base = next((cls for cls in type(obj).__mro__ if cls in _MODELS), None)
    if base is None:
        raise TypeError(f"Cannot convert object of type {type(obj).__name__!r}.")
    source, kind = _MODELS[base]
    if not hasattr(package, kind):
        raise ValueError(f"OME-Zarr v{version} has no equivalent of {kind!r}.")

    # (not excluding None, which would drop user-defined keys whose value is `null`)
    dumped = obj.model_dump(exclude_none=False)
    # (the fields of a subclass aren't part of the spec, and may clash with its keys)
    fields = {field.alias or name for name, field in base.model_fields.items()}
    doc = _strip_versions({k: v for k, v in dumped.items() if k in fields})
    start, stop = _ORDER.index(source), _ORDER.index(version)
    path = _ORDER[start : stop + 1] if start <= stop else _ORDER[stop : start + 1][::-1]
    for hop in zip(path, path[1:]):
        if step := _STEPS.get(hop):
            doc = step(doc, coordinate_system, dropped)
    return getattr(package, kind).model_validate(doc)


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

_Image: TypeAlias = "v04.Image | v05.Image | v06.Image"
_LabelImage: TypeAlias = "v04.LabelImage | v05.LabelImage | v06.LabelImage"
_LabelsGroup: TypeAlias = "v04.LabelsGroup | v05.LabelsGroup | v06.LabelsGroup"
_Plate: TypeAlias = "v04.Plate | v05.Plate | v06.Plate"
_Well: TypeAlias = "v04.Well | v05.Well | v06.Well"
_Bf2Raw: TypeAlias = "v04.Bf2Raw | v05.Bf2Raw | v06.Bf2Raw"
_Series: TypeAlias = "v04.Series | v05.Series | v06.Series"
_Multiscale: TypeAlias = "v04.Multiscale | v05.Multiscale | v06.Multiscale"
_Group: TypeAlias = "v05.OMEZarrGroupJSON | v06.OMEZarrGroupJSON"
_Attributes: TypeAlias = "v05.OMEAttributes | v06.OMEAttributes"
_04: TypeAlias = Literal["0.4"]
_05: TypeAlias = Literal["0.5"]
_06: TypeAlias = Literal["0.6"]


# fmt: off
# (LabelImage subclasses Image, so it must come first)
@overload
def convert_version(obj: _LabelImage, version: _04, *, coordinate_system: str = ...) -> v04.LabelImage: ...  # noqa: E501
@overload
def convert_version(obj: _LabelImage, version: _05, *, coordinate_system: str = ...) -> v05.LabelImage: ...  # noqa: E501
@overload
def convert_version(obj: _LabelImage, version: _06, *, coordinate_system: str = ...) -> v06.LabelImage: ...  # noqa: E501
@overload
def convert_version(obj: _Image, version: _04, *, coordinate_system: str = ...) -> v04.Image: ...  # noqa: E501
@overload
def convert_version(obj: _Image, version: _05, *, coordinate_system: str = ...) -> v05.Image: ...  # noqa: E501
@overload
def convert_version(obj: _Image, version: _06, *, coordinate_system: str = ...) -> v06.Image: ...  # noqa: E501
@overload
def convert_version(obj: _LabelsGroup, version: _04, *, coordinate_system: str = ...) -> v04.LabelsGroup: ...  # noqa: E501
@overload
def convert_version(obj: _LabelsGroup, version: _05, *, coordinate_system: str = ...) -> v05.LabelsGroup: ...  # noqa: E501
@overload
def convert_version(obj: _LabelsGroup, version: _06, *, coordinate_system: str = ...) -> v06.LabelsGroup: ...  # noqa: E501
@overload
def convert_version(obj: _Plate, version: _04, *, coordinate_system: str = ...) -> v04.Plate: ...  # noqa: E501
@overload
def convert_version(obj: _Plate, version: _05, *, coordinate_system: str = ...) -> v05.Plate: ...  # noqa: E501
@overload
def convert_version(obj: _Plate, version: _06, *, coordinate_system: str = ...) -> v06.Plate: ...  # noqa: E501
@overload
def convert_version(obj: _Well, version: _04, *, coordinate_system: str = ...) -> v04.Well: ...  # noqa: E501
@overload
def convert_version(obj: _Well, version: _05, *, coordinate_system: str = ...) -> v05.Well: ...  # noqa: E501
@overload
def convert_version(obj: _Well, version: _06, *, coordinate_system: str = ...) -> v06.Well: ...  # noqa: E501
@overload
def convert_version(obj: _Bf2Raw, version: _04, *, coordinate_system: str = ...) -> v04.Bf2Raw: ...  # noqa: E501
@overload
def convert_version(obj: _Bf2Raw, version: _05, *, coordinate_system: str = ...) -> v05.Bf2Raw: ...  # noqa: E501
@overload
def convert_version(obj: _Bf2Raw, version: _06, *, coordinate_system: str = ...) -> v06.Bf2Raw: ...  # noqa: E501
@overload
def convert_version(obj: _Series, version: _04, *, coordinate_system: str = ...) -> v04.Series: ...  # noqa: E501
@overload
def convert_version(obj: _Series, version: _05, *, coordinate_system: str = ...) -> v05.Series: ...  # noqa: E501
@overload
def convert_version(obj: _Series, version: _06, *, coordinate_system: str = ...) -> v06.Series: ...  # noqa: E501
@overload
def convert_version(obj: _Multiscale, version: _04, *, coordinate_system: str = ...) -> v04.Multiscale: ...  # noqa: E501
@overload
def convert_version(obj: _Multiscale, version: _05, *, coordinate_system: str = ...) -> v05.Multiscale: ...  # noqa: E501
@overload
def convert_version(obj: _Multiscale, version: _06, *, coordinate_system: str = ...) -> v06.Multiscale: ...  # noqa: E501
@overload
def convert_version(obj: v06.Scene, version: _06, *, coordinate_system: str = ...) -> v06.Scene: ...  # noqa: E501
@overload
def convert_version(obj: _Group | _Attributes, version: _04, *, coordinate_system: str = ...) -> v04.OMEZarrGroupJSON: ...  # noqa: E501
@overload
def convert_version(obj: _Group, version: _05, *, coordinate_system: str = ...) -> v05.OMEZarrGroupJSON: ...  # noqa: E501
@overload
def convert_version(obj: _Group, version: _06, *, coordinate_system: str = ...) -> v06.OMEZarrGroupJSON: ...  # noqa: E501
@overload
def convert_version(obj: _Attributes, version: _05, *, coordinate_system: str = ...) -> v05.OMEAttributes: ...  # noqa: E501
@overload
def convert_version(obj: _Attributes, version: _06, *, coordinate_system: str = ...) -> v06.OMEAttributes: ...  # noqa: E501
# fmt: on
def convert_version(
    obj: Any, version: Version, *, coordinate_system: str = "intrinsic"
) -> Any:
    """Convert an OME-Zarr metadata model to another version of the spec.

    Only metadata is converted: no store is read or written, and arrays are not
    touched.  (Note that v0.4 stores use Zarr v2, while later versions use Zarr v3.)

    Parameters
    ----------
    obj : Any
        The model to convert: an `Image`, `LabelImage`, `LabelsGroup`, `Plate`, `Well`,
        `Bf2Raw`, `Series`, `Multiscale` or `Scene` of any version, or an
        `OMEZarrGroupJSON` or `OMEAttributes`. It is not modified.
    version : "0.4" | "0.5" | "0.6"
        The version to convert to.  Converting to the same version returns a new,
        revalidated model (and normalizes a pre-release `version` like "0.6.dev4").
    coordinate_system : str, optional
        Name for the intrinsic coordinate system, when converting to v0.6 from an
        older version.  An existing coordinate system is never renamed.
        Default is "intrinsic".

    Returns
    -------
    Any
        The same kind of model, in the target version.  v0.4 has no `OMEZarrGroupJSON`
        or `OMEAttributes` model, so those convert to the metadata document they hold.

    Warns
    -----
    ConversionWarning
        If the target version can't express some of the metadata, which is dropped.
        Use `warnings.filterwarnings("error", category=ConversionWarning)` to raise
        instead.

    Raises
    ------
    TypeError
        If `obj` is not a model that can be converted.
    ValueError
        If `version` is unknown, or has no equivalent of `obj` (e.g. a v0.6 `Scene`).
    pydantic.ValidationError
        If the metadata is not valid in the target version (e.g. 6 axes in v0.5).

    Examples
    --------
    >>> from yaozarrs import DimSpec, convert_version, v04
    >>> dims = [DimSpec(name="y", size=64), DimSpec(name="x", size=64)]
    >>> image = v04.Image(multiscales=[v04.Multiscale.from_dims(dims)])
    >>> convert_version(image, "0.6").version
    '0.6'
    """
    if version not in _PACKAGES:
        raise ValueError(f"Unknown version {version!r}, expected one of {_ORDER}.")

    dropped: list[str] = []
    result = _convert(obj, version, coordinate_system, dropped)
    if dropped:
        warnings.warn(
            f"Converting to OME-Zarr v{version} dropped metadata that it can't "
            f"express: {', '.join(dict.fromkeys(dropped))}.",
            ConversionWarning,
            stacklevel=2,
        )
    return result
