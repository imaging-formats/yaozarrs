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


def scale_vec_ds(path: str, scale: list[float], out: str = "cs") -> dict:
    return {
        "path": path,
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": scale,
                "input": {"path": path},
                "output": {"name": out},
            }
        ],
    }


def identity_ds(path: str, out: str = "cs") -> dict:
    return {
        "path": path,
        "coordinateTransformations": [
            {"type": "identity", "input": {"path": path}, "output": {"name": out}}
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
    # typed space axes without units
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [
                    cs(
                        "phys",
                        [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
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
                "datasets": [identity_ds("0")],
            }
        ],
    },
    # identity (== scale of ones) as the finest level, then a coarser scaled level
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [identity_ds("0"), scale_vec_ds("1", [2.0, 2.0])],
            }
        ],
    },
    # anisotropic but element-wise non-decreasing ordering is valid
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [
                    scale_vec_ds("0", [1.0, 1.0]),
                    scale_vec_ds("1", [1.0, 2.0]),
                ],
            }
        ],
    },
    # multiscale-level coordinateTransformations: input=intrinsic, output=another
    # declared coordinate system
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [
                    cs("cs", [X_AXIS, Y_AXIS]),
                    cs("phys", [X_AXIS, Y_AXIS]),
                ],
                "datasets": [scale_ds("0", 2, out="cs")],
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [2.0, 2.0],
                        "input": {"name": "cs"},
                        "output": {"name": "phys"},
                    }
                ],
            }
        ],
    },
    # multiscale-level coordinateTransformations: link to a child labels group
    # (output has name + path) with an allowed identity transform
    {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [scale_ds("0", 2, out="cs")],
                "coordinateTransformations": [
                    {
                        "type": "identity",
                        "input": {"name": "cs"},
                        "output": {"name": "cs", "path": "labels/foo"},
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
        "either 2-3 axes of type 'space'",
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
    # datasets output to *different* (declared) coordinate systems
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [
                        cs("cs", [X_AXIS, Y_AXIS]),
                        cs("other", [X_AXIS, Y_AXIS]),
                    ],
                    "datasets": [
                        scale_ds("0", 2, out="cs", factor=1.0),
                        scale_ds("1", 2, out="other", factor=2.0),
                    ],
                }
            ],
        },
        "must output to the same coordinate system",
    ),
    # wrong resolution ordering (ordering check is independent of axis types)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [
                        scale_vec_ds("0", [2.0, 2.0]),
                        scale_vec_ds("1", [1.0, 1.0]),
                    ],
                }
            ],
        },
        "not ordered from highest to lowest resolution",
    ),
    # anisotropic, lexicographically "sorted" but element-wise decreasing (dim 2
    # goes 4 -> 1): must be rejected
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [
                        scale_vec_ds("0", [1.0, 4.0]),
                        scale_vec_ds("1", [2.0, 1.0]),
                    ],
                }
            ],
        },
        "not ordered from highest to lowest resolution",
    ),
    # identity (scale of ones) after a coarser scaled level: identity is finer
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_vec_ds("0", [2.0, 2.0]), identity_ds("1")],
                }
            ],
        },
        "not ordered from highest to lowest resolution",
    ),
    # two coordinate systems share a name (differ in axes)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [
                        cs("cs", [X_AXIS, Y_AXIS]),
                        cs("cs", [X_AXIS, Y_AXIS, Z_AXIS]),
                    ],
                    "datasets": [scale_ds("0", 2)],
                }
            ],
        },
        "names must be unique",
    ),
    # multiscale-level CT: input is not the intrinsic coordinate system
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [
                        cs("cs", [X_AXIS, Y_AXIS]),
                        cs("phys", [X_AXIS, Y_AXIS]),
                    ],
                    "datasets": [scale_ds("0", 2, out="cs")],
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [2.0, 2.0],
                            "input": {"name": "phys"},
                            "output": {"name": "cs"},
                        }
                    ],
                }
            ],
        },
        "must be the intrinsic coordinate system",
    ),
    # multiscale-level CT: output references an undeclared coordinate system
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 2, out="cs")],
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [2.0, 2.0],
                            "input": {"name": "cs"},
                            "output": {"name": "nope"},
                        }
                    ],
                }
            ],
        },
        "is not declared in coordinateSystems",
    ),
    # multiscale-level CT: labels link (output has path) with a disallowed type
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 2, out="cs")],
                    "coordinateTransformations": [
                        {
                            "type": "affine",
                            "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            "input": {"name": "cs"},
                            "output": {"name": "cs", "path": "labels/foo"},
                        }
                    ],
                }
            ],
        },
        "must be an identity, scale, or translation",
    ),
    # multiscale-level CT: missing input
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                    "datasets": [scale_ds("0", 2, out="cs")],
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [2.0, 2.0],
                            "output": {"name": "cs"},
                        }
                    ],
                }
            ],
        },
        "'input' must reference the intrinsic",
    ),
    # type-less axes are invalid (fail the axes.schema oneOf: 0 space, 0 array)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [
                        cs("cs", [{"name": "y"}, {"name": "x"}]),
                    ],
                    "datasets": [scale_vec_ds("0", [1.0, 1.0])],
                }
            ],
        },
        "either 2-3 axes of type 'space'",
    ),
    # a single axis can never satisfy the oneOf (needs >=2 space or >=2 array)
    (
        {
            "version": "0.6.dev4",
            "multiscales": [
                {
                    "coordinateSystems": [cs("cs", [{"name": "x", "type": "space"}])],
                    "datasets": [scale_vec_ds("0", [1.0])],
                }
            ],
        },
        "either 2-3 axes of type 'space'",
    ),
]


@pytest.mark.parametrize("obj, msg", V06_INVALID_IMAGES)
def test_invalid_v06_images(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v06.Image.model_validate(obj)


def test_should_warnings_v06() -> None:
    from yaozarrs._validation_warning import ValidationWarning

    # dataset transform: input SHOULD omit 'name'
    obj = {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0],
                                "input": {"path": "0", "name": "cs"},
                                "output": {"name": "cs"},
                            }
                        ],
                    }
                ],
            }
        ],
    }
    with pytest.warns(ValidationWarning, match="'input' SHOULD omit 'name'"):
        v06.Image.model_validate(obj)

    # dataset transform: output SHOULD omit 'path'
    obj2 = {
        "version": "0.6.dev4",
        "multiscales": [
            {
                "coordinateSystems": [cs("cs", [X_AXIS, Y_AXIS])],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0],
                                "input": {"path": "0"},
                                "output": {"name": "cs", "path": "somewhere"},
                            }
                        ],
                    }
                ],
            }
        ],
    }
    with pytest.warns(ValidationWarning, match="'output' SHOULD omit 'path'"):
        v06.Image.model_validate(obj2)


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
