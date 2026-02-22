import pytest
from pydantic import ValidationError

from yaozarrs import DimSpec, v04, validate_ome_object

X_AXIS = {"name": "x", "type": "space", "unit": "millimeter"}
Y_AXIS = {"name": "y", "type": "space", "unit": None}
Z_AXIS = {"name": "z", "type": "space", "unit": None}
T_AXIS = {"name": "t", "type": "time", "unit": "second"}
C_AXIS = {"name": "c", "type": "channel"}


def scale_tform(d: int) -> dict:
    return {"type": "scale", "scale": [1] * d}


def translate_tform(d: int, offset: float = 10.0) -> dict:
    return {"type": "translation", "translation": [offset] * d}


V04_VALID_IMAGES = [
    {
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
    # Test with time axis
    {
        "multiscales": [
            {
                "name": "timeseries",
                "axes": [T_AXIS, X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
    },
    # Test with channel axis
    {
        "multiscales": [
            {
                "name": "multichannel",
                "axes": [C_AXIS, X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
    },
    # Test with all axes
    {
        "multiscales": [
            {
                "name": "full5d",
                "axes": [T_AXIS, C_AXIS, Z_AXIS, Y_AXIS, X_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(5)]},
                ],
            }
        ],
    },
    # Test with translation transformation
    {
        "multiscales": [
            {
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            scale_tform(2),
                            translate_tform(2),
                        ],
                    },
                ],
            }
        ],
    },
    # Test with omero metadata
    {
        "multiscales": [
            {
                "axes": [C_AXIS, Y_AXIS, X_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
        "omero": {
            "channels": [
                {
                    "window": {"start": 0, "min": 0, "end": 100, "max": 255},
                    "color": "FF0000",
                }
            ]
        },
    },
]


V04_INVALID_IMAGES = [
    # Missing required multiscales
    {},
    # Empty multiscales array
    {"multiscales": []},
    # Missing axes
    {
        "multiscales": [
            {
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                ]
            }
        ]
    },
    # Insufficient space axes (only 1)
    {
        "multiscales": [
            {
                "axes": [X_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(1)]},
                ],
            }
        ]
    },
    # Too many space axes (4)
    {
        "multiscales": [
            {
                "axes": [X_AXIS, Y_AXIS, Z_AXIS, {"name": "w", "type": "space"}],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(4)]},
                ],
            }
        ]
    },
    # Multiple time axes
    {
        "multiscales": [
            {
                "axes": [T_AXIS, {"name": "t2", "type": "time"}, X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(4)]},
                ],
            }
        ]
    },
    # Multiple channel axes
    {
        "multiscales": [
            {
                "axes": [C_AXIS, {"name": "c2", "type": "channel"}, X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(4)]},
                ],
            }
        ]
    },
    # No scale transformation
    {
        "multiscales": [
            {
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [translate_tform(2)]},
                ],
            }
        ]
    },
    # Multiple scale transformations
    {
        "multiscales": [
            {
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [scale_tform(2), scale_tform(2)],
                    },
                ],
            }
        ]
    },
    # Mismatched transformation dimensions
    {
        "multiscales": [
            {
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ]
    },
    # Empty datasets
    {
        "multiscales": [
            {
                "axes": [X_AXIS, Y_AXIS],
                "datasets": [],
            }
        ]
    },
    # Duplicate axis names
    {
        "multiscales": [
            {
                "axes": [X_AXIS, X_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                ],
            }
        ]
    },
]


@pytest.mark.parametrize("data", V04_VALID_IMAGES)
def test_valid_v04_images(data: dict) -> None:
    """Test that valid v04 image metadata can be parsed."""
    img = v04.Image.model_validate(data)
    assert img.multiscales
    assert len(img.multiscales) >= 1


# Extended invalid cases with specific error messages
V04_INVALID_IMAGES_EXTENDED: list[tuple[dict, str]] = [
    # Multiple translation transforms (should fail)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "test",
                            "coordinateTransformations": [
                                scale_tform(2),
                                translate_tform(2),
                                translate_tform(2),  # Second translation
                            ],
                        }
                    ],
                }
            ]
        },
        "at most one translation transformation",
    ),
    # Translation before scale (should fail)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "test",
                            "coordinateTransformations": [
                                translate_tform(2),  # Before scale
                                scale_tform(2),
                            ],
                        }
                    ],
                }
            ]
        },
        "must be listed after the scale transformation",
    ),
    # No scale transform (should fail)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "test",
                            "coordinateTransformations": [translate_tform(2)],
                        }
                    ],
                }
            ]
        },
        "exactly one scale transformation",
    ),
    # Empty transformations list (should fail)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [{"path": "test", "coordinateTransformations": []}],
                }
            ]
        },
        "at least 1 item",
    ),
    # Datasets with different dimensions (should fail)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                        {"path": "1", "coordinateTransformations": [scale_tform(3)]},
                    ],
                }
            ]
        },
        "same number of dimensions",
    ),
    # More than 5 dimensions (should fail)
    (
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                        {"name": "extra", "type": "space"},  # 6 dimensions
                    ],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(6)]}
                    ],
                }
            ]
        },
        "not have more than 5 dimensions",
    ),
    # Global transform dimensions don't match axes
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],  # 2D
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]}
                    ],
                    "coordinateTransformations": [scale_tform(3)],  # 3D transform
                }
            ]
        },
        "does not match the number of axes",
    ),
    # Axis names must be unique (duplicate names)
    (
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "x", "type": "space"},
                        {"name": "x", "type": "space"},  # Duplicate name
                    ],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]}
                    ],
                }
            ]
        },
        "List items are not unique",
    ),
    # Transformation with 1D should fail (below minimum)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0]}  # 1D
                            ],
                        }
                    ],
                }
            ]
        },
        "at least 2 items",
    ),
]

# Convert V04_INVALID_IMAGES to same format as v05, with better error patterns
V04_INVALID_IMAGES_WITH_MESSAGES: list[tuple[dict, str]] = [
    # Missing required multiscales
    ({}, "Field required"),
    # Empty multiscales array
    ({"multiscales": []}, "at least 1 item"),
    # Missing axes
    (
        {
            "multiscales": [
                {
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    ]
                }
            ]
        },
        "Field required",
    ),
    # Insufficient space axes (only 1)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(1)]},
                    ],
                }
            ]
        },
        "at least 2 items",
    ),
    # Too many space axes (4)
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS, Z_AXIS, {"name": "w", "type": "space"}],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(4)]},
                    ],
                }
            ]
        },
        "2 or 3 axes of type 'space'",
    ),
    # Multiple time axes
    (
        {
            "multiscales": [
                {
                    "axes": [T_AXIS, {"name": "t2", "type": "time"}, X_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(4)]},
                    ],
                }
            ]
        },
        "at most 1 axis of type 'time'",
    ),
    # Multiple channel axes
    (
        {
            "multiscales": [
                {
                    "axes": [C_AXIS, {"name": "c2", "type": "channel"}, X_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(4)]},
                    ],
                }
            ]
        },
        "at most 1 axis of type 'channel'",
    ),
    # No scale transformation
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [translate_tform(2)],
                        },
                    ],
                }
            ]
        },
        "exactly one scale transformation",
    ),
    # Multiple scale transformations
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                scale_tform(2),
                                scale_tform(2),
                            ],
                        },
                    ],
                }
            ]
        },
        "exactly one scale transformation",
    ),
    # Mismatched transformation dimensions
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                    ],
                }
            ]
        },
        "does not match the number of axes",
    ),
    # Empty datasets
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, Y_AXIS],
                    "datasets": [],
                }
            ]
        },
        "at least 1 item",
    ),
    # Duplicate axis names
    (
        {
            "multiscales": [
                {
                    "axes": [X_AXIS, X_AXIS],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [scale_tform(2)]},
                    ],
                }
            ]
        },
        "List items are not unique",
    ),
]

# Combine all invalid cases with specific error messages
V04_ALL_INVALID_IMAGES: list[tuple[dict, str]] = (
    V04_INVALID_IMAGES_WITH_MESSAGES + V04_INVALID_IMAGES_EXTENDED
)


@pytest.mark.parametrize("obj, msg", V04_ALL_INVALID_IMAGES)
def test_invalid_v04_images(obj: dict, msg: str) -> None:
    """Test that invalid v04 image metadata raises validation errors."""
    with pytest.raises(ValidationError, match=msg):
        v04.Image.model_validate(obj)


# Additional tests for positive cases that are not covered by parameterized tests
def test_v04_image_with_omero() -> None:
    """Test v04 image with OMERO metadata."""
    data = {
        "multiscales": [
            {
                "axes": [C_AXIS, Y_AXIS, X_AXIS],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [scale_tform(3)]},
                ],
            }
        ],
        "omero": {
            "channels": [
                {
                    "window": {"start": 0.0, "min": 0.0, "end": 100.0, "max": 255.0},
                    "color": "FF0000",
                    "label": "Red Channel",
                    "active": True,
                }
            ]
        },
    }

    img = v04.Image.model_validate(data)
    assert img.omero is not None
    assert len(img.omero.channels) == 1
    assert img.omero.channels[0].color == "FF0000"


def test_v04_axis_types() -> None:
    """Test all axis types and their properties."""
    # Test SpaceAxis with all valid units
    space_axis = v04.SpaceAxis(name="x", unit="micrometer")
    assert space_axis.name == "x"
    assert space_axis.type == "space"
    assert space_axis.unit == "micrometer"

    # Test SpaceAxis without unit
    space_axis_no_unit = v04.SpaceAxis(name="y")
    assert space_axis_no_unit.unit is None

    # Test TimeAxis with valid units
    time_axis = v04.TimeAxis(name="t", unit="second")
    assert time_axis.name == "t"
    assert time_axis.type == "time"
    assert time_axis.unit == "second"

    # Test ChannelAxis
    channel_axis = v04.ChannelAxis(name="c")
    assert channel_axis.name == "c"
    assert channel_axis.type == "channel"

    # Test CustomAxis with type
    custom_axis = v04.CustomAxis(name="custom", type="custom_type")
    assert custom_axis.name == "custom"
    assert custom_axis.type == "custom_type"

    # Test CustomAxis without type
    custom_axis_no_type = v04.CustomAxis(name="other")
    assert custom_axis_no_type.type is None


def test_v04_transformations() -> None:
    """Test transformation objects and their properties."""
    # Test ScaleTransformation
    scale = v04.ScaleTransformation(scale=[1.0, 2.0, 3.0])
    assert scale.type == "scale"
    assert scale.scale == [1.0, 2.0, 3.0]
    assert scale.ndim == 3

    # Test TranslationTransformation
    translation = v04.TranslationTransformation(translation=[10.0, 20.0])
    assert translation.type == "translation"
    assert translation.translation == [10.0, 20.0]
    assert translation.ndim == 2


def test_v04_dataset_model() -> None:
    """Test Dataset model directly."""
    dataset = v04.Dataset(
        path="level_0",
        coordinateTransformations=[
            v04.ScaleTransformation(scale=[1.0, 1.0]),
            v04.TranslationTransformation(translation=[0.0, 0.0]),
        ],
    )
    assert dataset.path == "level_0"
    assert len(dataset.coordinateTransformations) == 2
    assert isinstance(dataset.coordinateTransformations[0], v04.ScaleTransformation)
    assert isinstance(
        dataset.coordinateTransformations[1], v04.TranslationTransformation
    )


def test_v04_omero_models() -> None:
    """Test OMERO-related models directly."""
    # Test OmeroWindow
    window = v04.OmeroWindow(start=0.0, min=0.0, end=100.0, max=255.0)
    assert window.start == 0.0
    assert window.min == 0.0
    assert window.end == 100.0
    assert window.max == 255.0

    # Test OmeroChannel with all fields
    channel = v04.OmeroChannel(
        window=window, color="FF0000", label="Red", family="linear", active=True
    )
    assert channel.window == window
    assert channel.color == "FF0000"
    assert channel.label == "Red"
    assert channel.family == "linear"
    assert channel.active is True

    # Test OmeroChannel with minimal fields
    minimal_channel = v04.OmeroChannel(window=window, color="00FF00")
    assert minimal_channel.color == "00FF00"
    assert minimal_channel.label is None
    assert minimal_channel.family is None
    assert minimal_channel.active is None

    # Test Omero
    omero = v04.Omero(channels=[channel, minimal_channel])
    assert len(omero.channels) == 2


def test_v04_axis_name_uniqueness_custom_validation() -> None:
    """Test that axis names must be unique - triggers custom validation logic."""
    # Import the validation function directly to test it
    from yaozarrs._axis import _validate_axes_list

    # Create axes with same names but different types to bypass UniqueList
    axes = [
        v04.SpaceAxis(name="duplicate"),
        v04.TimeAxis(name="duplicate"),  # Same name, different types
        v04.SpaceAxis(name="y"),
    ]

    # This should trigger our custom validation
    with pytest.raises(
        ValueError, match=r"Axis names must be unique\. Found duplicates"
    ):
        _validate_axes_list(axes)


def test_multiscale_from_dims() -> None:
    """Test Multiscale.from_dims with all features."""
    dims = [
        DimSpec(name="t", size=10, scale=1.0, unit="second"),
        DimSpec(name="c", size=3),
        # z with custom scale_factor=1.0 (no downsampling)
        DimSpec(name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0),
        DimSpec(name="y", size=512, scale=0.5, unit="micrometer"),
        DimSpec(name="x", size=512, scale=0.5, unit="micrometer"),
    ]
    ms = v04.Multiscale.from_dims(dims, name="test_image", n_levels=3)

    # Basic structure
    assert ms.name == "test_image"
    assert ms.ndim == 5
    assert ms.version == "0.4"
    assert len(ms.datasets) == 3

    # Axes: names, types
    assert [ax.name for ax in ms.axes] == ["t", "c", "z", "y", "x"]
    assert [ax.type for ax in ms.axes] == ["time", "channel", "space", "space", "space"]

    # Pyramid scales: t/c don't scale, z doesn't scale (factor=1.0), xy scale by 2x
    assert ms.datasets[0].coordinateTransformations[0].scale == [
        1.0,
        1.0,
        2.0,
        0.5,
        0.5,
    ]
    assert ms.datasets[1].coordinateTransformations[0].scale == [
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
    ]
    assert ms.datasets[2].coordinateTransformations[0].scale == [
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
    ]

    # Creates valid Image
    img = v04.Image(multiscales=[ms])
    validate_ome_object(img)


def test_dataset_path_warns_on_risky_characters() -> None:
    """Test that Dataset.path warns on characters outside [A-Za-z0-9._-]."""
    # Path with risky character (space) should warn
    with pytest.warns(UserWarning, match=r"Dataset\.path.*risky characters"):
        v04.Dataset(
            path="level 0",  # space is risky
            coordinateTransformations=[v04.ScaleTransformation(scale=[1.0, 1.0])],
        )

    # Path with safe characters should not warn (warnings are errors by default)
    v04.Dataset(
        path="level_0.test-1",  # all safe: alphanumeric, underscore, dot, hyphen
        coordinateTransformations=[v04.ScaleTransformation(scale=[1.0, 1.0])],
    )
