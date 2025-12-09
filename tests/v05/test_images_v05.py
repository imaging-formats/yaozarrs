import pytest
from pydantic import ValidationError

from yaozarrs import DimSpec, v05, validate_ome_object

X_AXIS = {"name": "x", "type": "space", "unit": "millimeter"}
Y_AXIS = {"name": "y", "type": "space", "unit": None}
Z_AXIS = {"name": "z", "type": "space", "unit": None}
T_AXIS = {"name": "t", "type": "time", "unit": "second"}
C_AXIS = {"name": "c", "type": "channel", "unit": None}


def scale_tform(d: int) -> dict:
    return {"type": "scale", "scale": [1] * d}


def translate_tform(d: int, offset: float = 10.0) -> dict:
    return {"type": "translation", "translation": [offset] * d}


V05_VALID_IMAGES = [
    {
        "version": "0.5",
        "multiscales": [
            {
                "name": "image",
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    {"path": "1", "coordinateTransformations": [scale_tform(2)]},
                ],
            }
        ],
    },
    # Test with 3D space axes
    {
        "version": "0.5",
        "multiscales": [
            {
                "name": "image3d",
                "axes": [X_AXIS, Y_AXIS, Z_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
    },
    # Test with time axis (properly ordered)
    {
        "version": "0.5",
        "multiscales": [
            {
                "name": "timelapse",
                "axes": [T_AXIS, X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
    },
    # Test with channel axis (properly ordered)
    {
        "version": "0.5",
        "multiscales": [
            {
                "name": "multichannel",
                "axes": [
                    {"name": "c", "type": "channel", "unit": None},
                    X_AXIS,
                    Y_AXIS,
                ],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
    },
    # Test with custom axis (properly ordered - custom axes treated as channel-like)
    {
        "version": "0.5",
        "multiscales": [
            {
                "name": "custom",
                "axes": [
                    {"name": "custom", "type": None, "unit": "arb"},
                    X_AXIS,
                    Y_AXIS,
                ],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
    },
    # Test with translation transformation (after scale)
    {
        "version": "0.5",
        "multiscales": [
            {
                "name": "translated",
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            scale_tform(2),
                            translate_tform(2, 10.0),
                        ],
                    },
                ],
            }
        ],
    },
]


@pytest.mark.parametrize("obj", V05_VALID_IMAGES)
def test_valid_v05_images(obj: dict) -> None:
    validate_ome_object(obj, v05.Image)


# list of (invalid_obj, error_msg_substring)
V05_INVALID_IMAGES: list[tuple[dict, str]] = [
    # Test duplicate axis names (caught by UniqueList validation)
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, X_AXIS],  # duplicate names
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]}
                    ],
                }
            ],
        },
        "List items are not unique",
    ),
    # Test different axes with same names (caught by custom name validation)
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "x", "type": "space", "unit": None},
                        # same name, different unit
                        {"name": "x", "type": "space", "unit": "meter"},
                    ],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]}
                    ],
                }
            ],
        },
        "Axis names must be unique",
    ),
    # Test less than 2 space axes
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [T_AXIS, C_AXIS],  # no space axes
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(1)]}
                    ],
                }
            ],
        },
        "There must be 2 or 3 axes of type 'space'",
    ),
    # Test more than 3 space axes
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        X_AXIS,
                        Y_AXIS,
                        Z_AXIS,
                        {"name": "w", "type": "space", "unit": None},  # 4 space axes
                    ],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(4)]}
                    ],
                }
            ],
        },
        "There must be 2 or 3 axes of type 'space'",
    ),
    # Test multiple time axes
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "t1", "type": "time", "unit": "second"},  # 2 time axes
                        T_AXIS,
                        X_AXIS,
                        Y_AXIS,
                    ],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(4)]}
                    ],
                }
            ],
        },
        "There can be at most 1 axis of type 'time'",
    ),
    # Test multiple channel axes
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "c1", "type": "channel", "unit": None},  # 2 channels
                        C_AXIS,
                        X_AXIS,
                        Y_AXIS,
                    ],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(4)]}
                    ],
                }
            ],
        },
        "There can be at most 1 axis of type 'channel'",
    ),
    # Test incorrect axis ordering (space before time)
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    # space before time
                    "axes": [X_AXIS, T_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(3)]}
                    ],
                }
            ],
        },
        "Axes are not in the required order by type",
    ),
    # Test incorrect axis ordering (space before channel)
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    # space before channel
                    "axes": [X_AXIS, C_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(3)]}
                    ],
                }
            ],
        },
        "Axes are not in the required order by type",
    ),
    # Test no scale transformation
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            # no scale
                            "coordinateTransformations": [translate_tform(2)],
                        }
                    ],
                }
            ],
        },
        "There must be exactly one scale transformation",
    ),
    # Test multiple scale transformations
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]},
                                {"type": "scale", "scale": [2.0, 2.0]},  # 2 scales
                            ],
                        }
                    ],
                }
            ],
        },
        "There must be exactly one scale transformation",
    ),
    # Test multiple translation transformations
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            # 2 translations
                            "coordinateTransformations": [
                                scale_tform(2),
                                translate_tform(2, 10.0),
                                translate_tform(2, 20.0),
                            ],
                        }
                    ],
                }
            ],
        },
        "There can be at most one translation transformation",
    ),
    # Test translation before scale
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            # translation first
                            "coordinateTransformations": [
                                translate_tform(2),
                                scale_tform(2),
                            ],
                        }
                    ],
                }
            ],
        },
        "translation transformation is given, it must be listed after the scale",
    ),
    # Test datasets with different dimensions
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [scale_tform(2)],
                        },
                        {
                            "path": "1",
                            # different dims
                            "coordinateTransformations": [scale_tform(3)],
                        },
                    ],
                }
            ],
        },
        "All datasets must have the same number of dimensions",
    ),
    # Test dataset with more than 5 dimensions
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [scale_tform(6)],
                        },  # 6 dims > 5
                    ],
                }
            ],
        },
        "Datasets must not have more than 5 dimensions",
    ),
    # Test transformation dimension mismatch with axes
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],  # 2 axes
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [scale_tform(3)],
                        },  # 3 dims
                    ],
                }
            ],
        },
        r"The length of the transformation \(3\) does not match the number of axes",
    ),
    # Test coordinateTransformations at multiscale level with dimension mismatch
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],  # 2 axes
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    ],
                    "coordinateTransformations": [scale_tform(3)],  # 3 dims
                }
            ],
        },
        r"The length of the transformation \(3\) does not match the number of axes",
    ),
    # Test duplicate datasets (UniqueList violation)
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        # duplicates
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    ],
                }
            ],
        },
        "List items are not unique",
    ),
    # Test duplicate multiscales (UniqueList violation)
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "name": "image",
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    ],
                },
                {
                    "name": "image",
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    ],
                },  # duplicate
            ],
        },
        "List items are not unique",
    ),
    # Test datasets not ordered from highest to lowest resolution
    (
        {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [2.0, 2.0]}
                            ],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]}
                            ],
                        },
                    ],
                }
            ],
        },
        "The datasets are not ordered from highest to lowest resolution",
    ),
]


@pytest.mark.parametrize("obj, msg", V05_INVALID_IMAGES)
def test_invalid_v05_images(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v05.Image.model_validate(obj)


def test_multiscale_from_dims() -> None:
    """Test Multiscale.from_dims with all features."""
    dims = [
        DimSpec(name="t", size=10, scale=1.0, unit="second"),
        DimSpec(name="c", type="channel", size=3),
        # z with custom scale_factor=1.0 (no downsampling)
        DimSpec(name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0),
        DimSpec(name="y", size=512, scale=0.5, unit="micrometer", translation=20.0),
        DimSpec(name="x", size=512, scale=0.5, unit="micrometer", translation=30.0),
    ]
    ms = v05.Multiscale.from_dims(dims, name="test_image", n_levels=3)

    # Basic structure
    assert ms.name == "test_image"
    assert ms.ndim == 5
    assert len(ms.datasets) == 3

    # Axes: names, types, units
    assert [ax.name for ax in ms.axes] == ["t", "c", "z", "y", "x"]
    assert [ax.type for ax in ms.axes] == ["time", "channel", "space", "space", "space"]
    assert ms.axes[0].unit == "second"
    assert ms.axes[1].unit is None
    assert ms.axes[2].unit == "micrometer"

    # Pyramid scales: t/c don't scale, z doesn't scale (factor=1.0), xy scale by 2x

    assert ms.datasets[0].scale_transform.scale == [1.0, 1.0, 2.0, 0.5, 0.5]
    assert ms.datasets[1].scale_transform.scale == [1.0, 1.0, 2.0, 1.0, 1.0]
    assert ms.datasets[2].scale_transform.scale == [1.0, 1.0, 2.0, 2.0, 2.0]

    for ds in ms.datasets:
        trans_tform = next(
            t for t in ds.coordinateTransformations if t.type == "translation"
        )
        assert trans_tform.translation == [0, 0, 0, 20.0, 30.0]

    # Creates valid Image
    img = v05.Image(multiscales=[ms])
    assert img.version == "0.5"
    validate_ome_object(img)
