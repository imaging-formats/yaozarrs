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
    # Check if zarr v3 is available for v0.5 tests
    import zarr

    zarr_major_version = int(zarr.__version__.split(".")[0])
    if zarr_major_version < 3:
        pytest.skip(
            reason="OME-ZARR v0.5 tests require zarr v3+, "
            "but zarr v{zarr.__version__} is installed",
            allow_module_level=True,
        )

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

    def test_extract_group_uri_with_zarr_json(self):
        """Test URI extraction with zarr.json paths specific to v0.5."""
        from yaozarrs.v05._storage import _StorageValidator

        # Test zarr.json path
        uri_zarr_json = "/data/test.zarr/zarr.json"
        extracted = _StorageValidator._extract_group_uri(uri_zarr_json)
        assert extracted == "/data/test.zarr"

        # Test Windows zarr.json path
        uri_windows = r"C:\data\test.zarr\zarr.json"
        extracted = _StorageValidator._extract_group_uri(uri_windows)
        assert extracted == r"C:\data\test.zarr"

        # Test URI without zarr.json suffix
        uri_no_suffix = "/data/test.zarr"
        extracted = _StorageValidator._extract_group_uri(uri_no_suffix)
        assert extracted == "/data/test.zarr"

    def test_validate_v05_version_metadata(self):
        """Test v0.5 specific version metadata validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Test missing OME version
            zarr_json_path = test_path / "zarr.json"
            invalid_data = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        # Missing version
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": [
                                    {"name": "x", "type": "space"},
                                    {"name": "y", "type": "space"},
                                ],
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
                    }
                },
            }

            with zarr_json_path.open("w") as f:
                json.dump(invalid_data, f)

            # Create a valid model but use invalid file
            valid_data = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": [
                                    {"name": "x", "type": "space"},
                                    {"name": "y", "type": "space"},
                                ],
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
                    }
                },
            }
            model = v05.OMEZarrGroupJSON.model_validate(valid_data)
            model.uri = str(zarr_json_path)

            result = v05.validate_storage(model)
            assert not result.valid
            error_messages = [error.message for error in result.errors]
            assert any("OME version must be '0.5'" in msg for msg in error_messages)

    def test_validate_zarr_format_warnings(self):
        """Test zarr format validation warnings for v0.5."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            zarr_json_path = test_path / "zarr.json"
            # Use zarr format 2 with v0.5 (should generate warning)
            data_with_zarr_v2 = {
                "zarr_format": 2,  # Should be 3 for v0.5
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": [
                                    {"name": "x", "type": "space"},
                                    {"name": "y", "type": "space"},
                                ],
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
                    }
                },
            }

            with zarr_json_path.open("w") as f:
                json.dump(data_with_zarr_v2, f)

            # Create valid model for pydantic validation
            valid_data = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": [
                                    {"name": "x", "type": "space"},
                                    {"name": "y", "type": "space"},
                                ],
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
                    }
                },
            }
            model = v05.OMEZarrGroupJSON.model_validate(valid_data)
            model.uri = str(zarr_json_path)

            result = v05.validate_storage(model)
            # Should be invalid due to format mismatch or generate warnings
            warning_messages = (
                [warning.message for warning in result.warnings]
                if result.warnings
                else []
            )
            # Check if zarr format validation produces appropriate feedback
            assert len(result.errors) > 0 or len(warning_messages) > 0

    def test_validate_multiscale_axes_required_v05(self):
        """Test that axes are required in v0.5 multiscales."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            zarr_json_path = test_path / "zarr.json"
            # Data without axes (invalid for v0.5)
            invalid_data = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                # Missing axes - required in v0.5
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
                    }
                },
            }

            with zarr_json_path.open("w") as f:
                json.dump(invalid_data, f)

            # Create valid model for validation
            valid_data = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "name": "test",
                                "axes": [
                                    {"name": "x", "type": "space"},
                                    {"name": "y", "type": "space"},
                                ],
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
                    }
                },
            }
            model = v05.OMEZarrGroupJSON.model_validate(valid_data)
            model.uri = str(zarr_json_path)

            result = v05.validate_storage(model)
            assert not result.valid
            error_messages = [error.message for error in result.errors]
            assert any("axes are required" in msg for msg in error_messages)
