"""Tests for OME-ZARR v0.5 storage validation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from yaozarrs import v05

# Add project root to path for scripts import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.write_demo_zarr import (
        write_ome_image,
        write_ome_labels,
        write_ome_plate,
    )
except ImportError as e:
    pytest.skip(reason=f"Cannot test storage: {e}", allow_module_level=True)


class TestV05StorageValidation:
    """Test storage validation for OME-ZARR v0.5."""

    def test_validate_storage_requires_zarr(self):
        """Test that validate_storage fails gracefully without zarr."""
        # Create a mock model
        model = v05.OMEZarrGroupJSON(
            zarr_format=3,
            node_type="group",
            attributes=v05.OMEAttributes(
                ome=v05.Image(
                    version="0.5",
                    multiscales=[
                        v05.Multiscale(
                            name="test",
                            axes=[
                                v05.SpaceAxis(name="x", type="space"),
                                v05.SpaceAxis(name="y", type="space"),
                            ],
                            datasets=[
                                v05.Dataset(
                                    path="0",
                                    coordinateTransformations=[
                                        v05.ScaleTransformation(scale=[1.0, 1.0])
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ),
        )

        # This should work since zarr is importable
        # But if model has no uri, it should fail
        model.uri = None
        with pytest.raises(ValueError, match="Model must have a uri field"):
            v05.validate_storage(model)

    def test_validate_storage_http_url(self):
        """Test that HTTP URLs fail for nonexistent URLs."""
        model = v05.OMEZarrGroupJSON(
            zarr_format=3,
            node_type="group",
            attributes=v05.OMEAttributes(
                ome=v05.Image(
                    version="0.5",
                    multiscales=[
                        v05.Multiscale(
                            name="test",
                            axes=[
                                v05.SpaceAxis(name="x", type="space"),
                                v05.SpaceAxis(name="y", type="space"),
                            ],
                            datasets=[
                                v05.Dataset(
                                    path="0",
                                    coordinateTransformations=[
                                        v05.ScaleTransformation(scale=[1.0, 1.0])
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ),
            uri="https://example.com/test.zarr",
        )

        result = v05.validate_storage(model)
        assert not result.valid
        assert len(result.errors) == 1
        assert "Cannot open zarr group" in result.errors[0].message

    def test_validate_storage_nonexistent_path(self):
        """Test validation of nonexistent zarr group."""
        model = v05.OMEZarrGroupJSON(
            zarr_format=3,
            node_type="group",
            attributes=v05.OMEAttributes(
                ome=v05.Image(
                    version="0.5",
                    multiscales=[
                        v05.Multiscale(
                            name="test",
                            axes=[
                                v05.SpaceAxis(name="x", type="space"),
                                v05.SpaceAxis(name="y", type="space"),
                            ],
                            datasets=[
                                v05.Dataset(
                                    path="0",
                                    coordinateTransformations=[
                                        v05.ScaleTransformation(scale=[1.0, 1.0])
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ),
            uri="/nonexistent/path/test.zarr",
        )

        result = v05.validate_storage(model)
        assert not result.valid
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0].message

    def test_validate_valid_image_storage(self):
        """Test validation of a valid image storage structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid OME-ZARR v0.5 image
            image_path = Path(tmp_dir) / "test_image.zarr"
            write_ome_image(
                image_path, version="0.5", shape=(64, 64), axes="yx", num_levels=2
            )

            # Load the zarr.json to create a model
            zarr_json_path = image_path / "zarr.json"
            with zarr_json_path.open() as f:
                zarr_data = json.load(f)

            model = v05.OMEZarrGroupJSON.model_validate(zarr_data)
            model.uri = str(zarr_json_path)

            # Validate storage
            result = v05.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_validate_valid_plate_storage(self):
        """Test validation of a valid plate storage structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid OME-ZARR v0.5 plate
            plate_path = Path(tmp_dir) / "test_plate.zarr"
            write_ome_plate(
                plate_path,
                version="0.5",
                rows=["A", "B"],
                columns=["1", "2"],
                image_shape=(32, 32),
                fields_per_well=1,
                image_axes="yx",
            )

            # Load the zarr.json to create a model
            zarr_json_path = plate_path / "zarr.json"
            with zarr_json_path.open() as f:
                zarr_data = json.load(f)

            model = v05.OMEZarrGroupJSON.model_validate(zarr_data)
            model.uri = str(zarr_json_path)

            # Validate storage
            result = v05.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_validate_valid_labels_storage(self):
        """Test validation of a valid labels storage structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid OME-ZARR v0.5 labels
            labels_path = Path(tmp_dir) / "test_labels.zarr"
            write_ome_labels(
                labels_path, version="0.5", shape=(32, 32), axes="yx", num_levels=2
            )

            # Labels demo creates a nested structure - test the actual label image
            label_image_path = labels_path / "labels" / "labels" / "zarr.json"
            with label_image_path.open() as f:
                zarr_data = json.load(f)

            model = v05.OMEZarrGroupJSON.model_validate(zarr_data)
            model.uri = str(label_image_path)

            # Validate storage
            result = v05.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_validate_missing_zarr_json(self):
        """Test validation when zarr.json is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create directory without zarr.json
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            model = v05.OMEZarrGroupJSON(
                zarr_format=3,
                node_type="group",
                attributes=v05.OMEAttributes(
                    ome=v05.Image(
                        version="0.5",
                        multiscales=[
                            v05.Multiscale(
                                name="test",
                                axes=[
                                    v05.SpaceAxis(name="x", type="space"),
                                    v05.SpaceAxis(name="y", type="space"),
                                ],
                                datasets=[
                                    v05.Dataset(
                                        path="0",
                                        coordinateTransformations=[
                                            v05.ScaleTransformation(scale=[1.0, 1.0])
                                        ],
                                    )
                                ],
                            )
                        ],
                    )
                ),
                uri=str(test_path),
            )

            result = v05.validate_storage(model)
            assert not result.valid
            assert any(
                "Cannot open zarr group" in error.message for error in result.errors
            )

    # Note: test_validate_invalid_zarr_format removed because
    # pydantic models now properly validate zarr_format constraints at creation time

    def test_validate_missing_dataset_path(self):
        """Test validation when referenced dataset path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid image first
            image_path = Path(tmp_dir) / "test_image.zarr"
            write_ome_image(
                image_path, version="0.5", shape=(64, 64), axes="yx", num_levels=2
            )

            # Remove one of the resolution levels
            resolution_path = image_path / "1"
            if resolution_path.exists():
                import shutil

                shutil.rmtree(resolution_path)

            # Load the zarr.json to create a model
            zarr_json_path = image_path / "zarr.json"
            with zarr_json_path.open() as f:
                zarr_data = json.load(f)

            model = v05.OMEZarrGroupJSON.model_validate(zarr_data)
            model.uri = str(zarr_json_path)

            # Validate storage
            result = v05.validate_storage(model)
            assert not result.valid
            assert any(
                "Dataset path does not exist: 1" in error.message
                for error in result.errors
            )

    # Note: test_validate_invalid_ome_version removed because
    # pydantic models now properly validate version constraints at model creation time

    def test_validate_missing_coordinate_transformations(self):
        """Test validation when coordinate transformations are missing scale."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            zarr_json_path = test_path / "zarr.json"
            invalid_data = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": [{"name": "x", "type": "space"}],
                                "datasets": [
                                    {
                                        "path": "0",
                                        "coordinateTransformations": [
                                            {
                                                "type": "translation",
                                                "translation": [10.0],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                },
            }

            with zarr_json_path.open("w") as f:
                json.dump(invalid_data, f)

            # Create the dataset directory
            dataset_path = test_path / "0"
            dataset_path.mkdir()
            array_json_path = dataset_path / "zarr.json"
            array_data = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [64],
                "data_type": "uint16",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [32]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            }
            with array_json_path.open("w") as f:
                json.dump(array_data, f)

            model = v05.OMEZarrGroupJSON.model_validate(invalid_data)
            model.uri = str(zarr_json_path)

            result = v05.validate_storage(model)
            assert not result.valid
            assert any(
                "at least one 'scale' transformation is required" in error.message
                for error in result.errors
            )
