import pytest
from pydantic import ValidationError

from yaozarrs import DimSpec, v06, validate_ome_object

X_AXIS = {"name": "x", "type": "space", "unit": "millimeter"}
Y_AXIS = {"name": "y", "type": "space", "unit": None}
Z_AXIS = {"name": "z", "type": "space", "unit": None}
T_AXIS = {"name": "t", "type": "time", "unit": "second"}
C_AXIS = {"name": "c", "type": "channel", "unit": None}


def cs(name: str, axes: list[dict]) -> dict:
    return {"name": name, "axes": axes}


def scale_ds(path: str, d: int, out: str = "cs", factor: float = 1.0) -> dict:
    return {
        "path": path,
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [factor] * d,
                "input": {"path": path},
                "output": {"name": out},
            }
        ],
    }


def seq_ds(path: str, d: int, out: str = "cs", factor: float = 1.0) -> dict:
    return {
        "path": path,
        "coordinateTransformations": [
            {
                "type": "sequence",
                "input": {"path": path},
                "output": {"name": out},
                "transformations": [
                    {"type": "scale", "scale": [factor] * d},
                    {"type": "translation", "translation": [10.0] * d},
                ],
            }
        ],
    }


V06_VALID_IMAGES = [
    # 2D
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "name": "image",
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [
                    scale_ds("0", 2, factor=1.0),
                    scale_ds("1", 2, factor=2.0),
                ],
            }
        ],
    },
    # 3D space
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "name": "image3d",
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS, Z_AXIS])],
                "datasets": [scale_ds("0", 3)],
            }
        ],
    },
    # time + space, properly ordered
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [T_AXIS, X_AXIS, Y_AXIS])],
                "datasets": [scale_ds("0", 3)],
            }
        ],
    },
    # type-less axes (allowed in v0.6) -- like the label spec example
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [
                    cs(
                        "phys",
                        [{"name": "y", "unit": "um"}, {"name": "x", "unit": "um"}],
                    )
                ],
                "datasets": [scale_ds("0", 2, out="phys")],
            }
        ],
    },
    # sequence (scale + translation) dataset transform
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [seq_ds("0", 2)],
            }
        ],
    },
    # identity dataset transform
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "identity",
                                "input": {"path": "0"},
                                "output": {"name": "cs"},
                            }
                        ],
                    }
                ],
            }
        ],
    },
]


@pytest.mark.parametrize("obj", V06_VALID_IMAGES)
def test_valid_v06_images(obj: dict) -> None:
    validate_ome_object(obj, v06.Image)


V06_INVALID_IMAGES: list[tuple[dict, str]] = [
    # duplicate axis names
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, X_AXIS])],
                    "datasets": [scale_ds("0", 2)],
                }
            ],
        },
        "List items are not unique|Axis names must be unique",
    ),
    # >3 space axes (fully typed -> enforced)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [
                        cs(
                            "cs",
                            [X_AXIS, Y_AXIS, Z_AXIS, {"name": "w", "type": "space"}],
                        )
                    ],
                    "datasets": [scale_ds("0", 4)],
                }
            ],
        },
        "There must be 2 or 3 axes of type 'space'",
    ),
    # bad axis ordering (space before time)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, T_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 3)],
                }
            ],
        },
        "Axes are not in the required order by type",
    ),
    # scale must be > 0
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 2, factor=0.0)],
                }
            ],
        },
        "greater than 0",
    ),
    # dataset transform missing input.path / output.name
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]}
                            ],
                        }
                    ],
                }
            ],
        },
        "must provide a 'path'|must provide a 'name'",
    ),
    # dataset transform of a disallowed type (affine)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "affine",
                                    "affine": [[1, 0], [0, 1]],
                                    "input": {"path": "0"},
                                    "output": {"name": "cs"},
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        "scale.*identity.*sequence|sequence.*scale",
    ),
    # ndim mismatch between transform and coordinate system axes
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 3)],
                }
            ],
        },
        "does not match the number of axes",
    ),
    # dataset outputs to a non-existent coordinate system
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 2, out="nope")],
                }
            ],
        },
        "is not declared in coordinateSystems",
    ),
    # datasets not ordered highest -> lowest resolution
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [
                        scale_ds("0", 2, factor=2.0),
                        scale_ds("1", 2, factor=1.0),
                    ],
                }
            ],
        },
        "not ordered from highest to lowest resolution",
    ),
    # missing coordinateSystems entirely
    (
        {
            "version": "0.6.dev4",
            "multiscales": [{"datasets": [scale_ds("0", 2)]}],
        },
        "coordinateSystems",
    ),
]


@pytest.mark.parametrize("obj, msg", V06_INVALID_IMAGES)
def test_invalid_v06_images(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v06.Image.model_validate(obj)


def test_multiscale_from_dims() -> None:
    dims = [
        DimSpec(name="t", size=10, scale=1.0, unit="second"),
        DimSpec(name="c", type="channel", size=3),
        DimSpec(name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0),
        DimSpec(name="y", size=512, scale=0.5, unit="micrometer", translation=20.0),
        DimSpec(name="x", size=512, scale=0.5, unit="micrometer", translation=30.0),
    ]
    ms = v06.Multiscale.from_dims(dims, name="test_image", n_levels=3)

    assert ms.name == "test_image"
    assert ms.ndim == 5
    assert len(ms.datasets) == 3
    assert ms.intrinsic_coordinate_system.name == "intrinsic"

    # axes property returns the intrinsic coordinate system's axes
    assert [ax.name for ax in ms.axes] == ["t", "c", "z", "y", "x"]
    assert [ax.type for ax in ms.axes] == ["time", "channel", "space", "space", "space"]

    # pyramid scales: t/c/z don't downsample, xy by 2x
    assert ms.datasets[0].scale_transform.scale == [1.0, 1.0, 2.0, 0.5, 0.5]
    assert ms.datasets[2].scale_transform.scale == [1.0, 1.0, 2.0, 2.0, 2.0]

    # translation present -> dataset transform is a sequence
    for ds in ms.datasets:
        assert ds.translation_transform is not None
        assert ds.translation_transform.translation == [0, 0, 0, 20.0, 30.0]
        # input/output wired up
        assert ds.transform.input.path == ds.path
        assert ds.transform.output.name == "intrinsic"

    img = v06.Image(multiscales=[ms])
    assert img.version == "0.6.dev4"
    validate_ome_object(img)


def test_axes_compat_property_readonly() -> None:
    ms = v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])
    # axes accessor mirrors the intrinsic coordinate system
    assert [a.name for a in ms.axes] == ["y", "x"]
    assert ms.coordinateSystems[0].axes is ms.axes
