"""Tests for OME-ZARR v0.4 storage validation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from yaozarrs import v04

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


class TestV04StorageValidation:
    """Test storage validation for OME-ZARR v0.4."""

    def test_validate_storage_requires_zarr(self):
        """Test that validate_storage fails gracefully without zarr."""
        # Create a mock model
        model = v04.Image(
            multiscales=[
                v04.Multiscale(
                    name="test",
                    axes=[
                        v04.SpaceAxis(name="x", type="space"),
                        v04.SpaceAxis(name="y", type="space"),
                    ],
                    datasets=[
                        v04.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v04.ScaleTransformation(scale=[1.0, 1.0])
                            ],
                        )
                    ],
                )
            ],
        )

        # This should work since zarr is importable
        # But if model has no uri, it should fail
        model.uri = None
        with pytest.raises(ValueError, match="Model must have a uri field"):
            v04.validate_storage(model)

    def test_validate_storage_nonexistent_path(self):
        """Test validation of nonexistent zarr group."""
        model = v04.Image(
            multiscales=[
                v04.Multiscale(
                    version="0.4",
                    name="test",
                    axes=[
                        v04.SpaceAxis(name="x", type="space"),
                        v04.SpaceAxis(name="y", type="space"),
                    ],
                    datasets=[
                        v04.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v04.ScaleTransformation(scale=[1.0, 1.0])
                            ],
                        )
                    ],
                )
            ],
            uri="/nonexistent/path/test.zarr",
        )

        result = v04.validate_storage(model)
        assert not result.valid
        assert len(result.errors) == 1

    def test_validate_valid_image_storage(self):
        """Test validation of a valid image storage structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid OME-ZARR v0.4 image
            image_path = Path(tmp_dir) / "test_image.zarr"
            write_ome_image(
                image_path, version="0.4", shape=(64, 64), axes="yx", num_levels=2
            )

            # Load the .zattrs to create a model
            zattrs_path = image_path / ".zattrs"
            with zattrs_path.open() as f:
                attrs_data = json.load(f)

            model = v04.Image.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            # Validate storage
            result = v04.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_validate_valid_plate_storage(self):
        """Test validation of a valid plate storage structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid OME-ZARR v0.4 plate
            plate_path = Path(tmp_dir) / "test_plate.zarr"
            write_ome_plate(
                plate_path,
                version="0.4",
                rows=["A", "B"],
                columns=["1", "2"],
                image_shape=(32, 32),
                fields_per_well=1,
                image_axes="yx",
            )

            # Load the .zattrs to create a model
            zattrs_path = plate_path / ".zattrs"
            with zattrs_path.open() as f:
                attrs_data = json.load(f)

            model = v04.Plate.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            # Validate storage
            result = v04.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_validate_valid_labels_storage(self):
        """Test validation of a valid labels storage structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid OME-ZARR v0.4 labels
            labels_path = Path(tmp_dir) / "test_labels.zarr"
            write_ome_labels(
                labels_path, version="0.4", shape=(32, 32), axes="yx", num_levels=2
            )

            # Labels demo creates a nested structure - test the actual label image
            label_image_path = labels_path / "labels" / "labels" / ".zattrs"
            with label_image_path.open() as f:
                attrs_data = json.load(f)

            model = v04.LabelImage.model_validate(attrs_data)
            model.uri = str(label_image_path)

            # Validate storage
            result = v04.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_validate_missing_zgroup(self):
        """Test validation when .zgroup is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create directory without .zgroup
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Create .zattrs but no .zgroup
            zattrs_path = test_path / ".zattrs"
            attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
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
                ]
            }
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            model = v04.Image.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            result = v04.validate_storage(model)
            assert not result.valid
            assert any(
                "Cannot open zarr group" in error.message for error in result.errors
            )

    def test_validate_invalid_zarr_format(self):
        """Test validation with invalid zarr format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create directory with invalid .zgroup
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Create .zgroup with wrong format
            zgroup_path = test_path / ".zgroup"
            zgroup_data = {
                "zarr_format": 3  # Should be 2 for v0.4
            }
            with zgroup_path.open("w") as f:
                json.dump(zgroup_data, f)

            # Create .zattrs
            zattrs_path = test_path / ".zattrs"
            attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
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
                ]
            }
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            model = v04.Image.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            result = v04.validate_storage(model)
            assert not result.valid
            # With zarr v2, we get a different error when trying to open zarr v3 format
            assert any(
                "unsupported zarr format: 3" in error.message
                or "zarr_format must be 2" in error.message
                for error in result.errors
            )

    def test_validate_missing_dataset_path(self):
        """Test validation when referenced dataset path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid image first
            image_path = Path(tmp_dir) / "test_image.zarr"
            write_ome_image(
                image_path, version="0.4", shape=(64, 64), axes="yx", num_levels=2
            )

            # Remove one of the resolution levels
            resolution_path = image_path / "1"
            if resolution_path.exists():
                import shutil

                shutil.rmtree(resolution_path)

            # Load the .zattrs to create a model
            zattrs_path = image_path / ".zattrs"
            with zattrs_path.open() as f:
                attrs_data = json.load(f)

            model = v04.Image.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            # Validate storage
            result = v04.validate_storage(model)
            assert not result.valid
            assert any(
                "Dataset path does not exist: 1" in error.message
                for error in result.errors
            )

    def test_validate_missing_zarray_in_dataset(self):
        """Test validation when dataset is missing .zarray file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid image first
            image_path = Path(tmp_dir) / "test_image.zarr"
            write_ome_image(
                image_path, version="0.4", shape=(64, 64), axes="yx", num_levels=1
            )

            # Remove .zarray from resolution level 0
            zarray_path = image_path / "0" / ".zarray"
            if zarray_path.exists():
                zarray_path.unlink()

            # Load the .zattrs to create a model
            zattrs_path = image_path / ".zattrs"
            with zattrs_path.open() as f:
                attrs_data = json.load(f)

            model = v04.Image.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            # Validate storage
            result = v04.validate_storage(model)
            assert not result.valid
            assert any(
                "Dataset path does not exist" in error.message
                for error in result.errors
            )

    # Note: test_validate_inconsistent_ome_version removed because
    # pydantic models now properly validate version constraints at model creation time

    # Note: test_validate_missing_coordinate_transformations removed because
    # pydantic models now properly validate that scale transformations are required

    def test_validate_bioformats2raw_layout(self):
        """Test validation of bioformats2raw layout."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "converted.zarr"
            test_path.mkdir()

            # Create .zgroup
            zgroup_path = test_path / ".zgroup"
            zgroup_data = {"zarr_format": 2}
            with zgroup_path.open("w") as f:
                json.dump(zgroup_data, f)

            # Create .zattrs with bioformats2raw layout
            zattrs_path = test_path / ".zattrs"
            attrs_data = {"bioformats2raw.layout": 3}
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            # Create numbered directories (0/, 1/)
            for i in range(2):
                img_path = test_path / str(i)
                img_path.mkdir()

                # Create basic zarr group structure
                img_zgroup_path = img_path / ".zgroup"
                with img_zgroup_path.open("w") as f:
                    json.dump({"zarr_format": 2}, f)

            model = v04.Bf2Raw(**attrs_data)
            model.uri = str(zattrs_path)

            result = v04.validate_storage(model)
            assert result.valid
            assert len(result.errors) == 0

    def test_extract_group_uri_with_windows_paths(self):
        """Test URI extraction with Windows-style paths."""
        from yaozarrs.v04._storage import _StorageValidator

        # Test Windows path with backslash
        uri_windows = r"C:\data\test.zarr\.zattrs"
        extracted = _StorageValidator._extract_group_uri(uri_windows)
        assert extracted == r"C:\data\test.zarr"

        # Test normal Unix path
        uri_unix = "/data/test.zarr/.zattrs"
        extracted = _StorageValidator._extract_group_uri(uri_unix)
        assert extracted == "/data/test.zarr"

        # Test URI without .zattrs suffix
        uri_no_suffix = "/data/test.zarr"
        extracted = _StorageValidator._extract_group_uri(uri_no_suffix)
        assert extracted == "/data/test.zarr"

    def test_validate_coordinate_transform_length_mismatch(self):
        """Test validation when coordinate transforms have wrong length."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Create .zgroup
            zgroup_path = test_path / ".zgroup"
            with zgroup_path.open("w") as f:
                json.dump({"zarr_format": 2}, f)

            # Create .zattrs with valid pydantic data but storage-level issues
            # Manually create invalid coordinate transformations after validation
            zattrs_path = test_path / ".zattrs"
            attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
                        "name": "test",
                        "axes": [
                            {"name": "x", "type": "space"},
                            {"name": "y", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0]},
                                    {
                                        "type": "translation",
                                        "translation": [10.0, 20.0],
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            # Create dataset directory and .zarray
            dataset_path = test_path / "0"
            dataset_path.mkdir()
            zarray_path = dataset_path / ".zarray"
            zarray_data = {
                "shape": [64, 64],
                "chunks": [32, 32],
                "dtype": "<u2",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "zarr_format": 2,
            }
            with zarray_path.open("w") as f:
                json.dump(zarray_data, f)

            # Manually manipulate the attributes to introduce storage-level errors
            # This bypasses pydantic validation but tests storage validation
            attrs_data["multiscales"][0]["datasets"][0]["coordinateTransformations"] = [
                {"type": "scale", "scale": [1.0]},  # Wrong length - should be 2
                {
                    "type": "translation",
                    "translation": [10.0, 20.0, 30.0],
                },  # Wrong length
            ]

            # Write the modified data directly to test storage validation
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            # Use a valid model but with the modified URI
            valid_attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
                        "name": "test",
                        "axes": [
                            {"name": "x", "type": "space"},
                            {"name": "y", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0]},
                                ],
                            }
                        ],
                    }
                ]
            }
            model = v04.Image.model_validate(valid_attrs_data)
            model.uri = str(zattrs_path)

            result = v04.validate_storage(model)
            assert not result.valid

            # Check for specific error messages
            error_messages = [error.message for error in result.errors]
            assert any(
                "scale length (1) must match axes count (2)" in msg
                for msg in error_messages
            )
            assert any(
                "translation length (3) must match axes count (2)" in msg
                for msg in error_messages
            )

    def test_validate_multiscale_without_scale_transform(self):
        """Test validation when scale transformation is missing (warning for v0.4)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Create .zgroup
            zgroup_path = test_path / ".zgroup"
            with zgroup_path.open("w") as f:
                json.dump({"zarr_format": 2}, f)

            # Create dataset
            dataset_path = test_path / "0"
            dataset_path.mkdir()
            zarray_path = dataset_path / ".zarray"
            zarray_data = {
                "shape": [64, 64],
                "chunks": [32, 32],
                "dtype": "<u2",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "zarr_format": 2,
            }
            with zarray_path.open("w") as f:
                json.dump(zarray_data, f)

            # Create .zattrs with only translation transform - modify after validation
            zattrs_path = test_path / ".zattrs"
            attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
                        "name": "test",
                        "axes": [
                            {"name": "x", "type": "space"},
                            {"name": "y", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "translation", "translation": [10.0, 20.0]}
                                ],
                            }
                        ],
                    }
                ]
            }
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            # Create a valid model for pydantic validation
            valid_attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
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
                ]
            }
            model = v04.Image.model_validate(valid_attrs_data)
            model.uri = str(zattrs_path)

            result = v04.validate_storage(model)
            # Should be valid but with warnings for v0.4
            assert result.valid
            assert len(result.warnings) > 0
            warning_messages = [warning.message for warning in result.warnings]
            assert any(
                "at least one 'scale' transformation is required" in msg
                for msg in warning_messages
            )

    def test_validate_label_data_types(self):
        """Test validation of label image data types."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a label image with non-integer data type
            labels_path = Path(tmp_dir) / "test_labels.zarr"
            write_ome_labels(
                labels_path, version="0.4", shape=(32, 32), axes="yx", num_levels=1
            )

            # Change the data type to float (should cause error)
            zarray_path = labels_path / "labels" / "labels" / "0" / ".zarray"
            with zarray_path.open() as f:
                zarray_data = json.load(f)

            # Change dtype to float
            zarray_data["dtype"] = "<f4"
            with zarray_path.open("w") as f:
                json.dump(zarray_data, f)

            # Load and validate
            label_zattrs_path = labels_path / "labels" / "labels" / ".zattrs"
            with label_zattrs_path.open() as f:
                attrs_data = json.load(f)

            model = v04.LabelImage.model_validate(attrs_data)
            model.uri = str(label_zattrs_path)

            result = v04.validate_storage(model)
            assert not result.valid
            error_messages = [error.message for error in result.errors]
            assert any(
                "Label arrays must use integer data types" in msg
                for msg in error_messages
            )

    def test_validate_empty_attributes(self):
        """Test validation with empty attributes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Create .zgroup
            zgroup_path = test_path / ".zgroup"
            with zgroup_path.open("w") as f:
                json.dump({"zarr_format": 2}, f)

            # Create empty .zattrs
            zattrs_path = test_path / ".zattrs"
            with zattrs_path.open("w") as f:
                json.dump({}, f)

            # Create a minimal model that bypasses pydantic validation
            class MinimalModel:
                def __init__(self, uri):
                    self.uri = uri

            model = MinimalModel(str(zattrs_path))  # type: ignore

            result = v04.validate_storage(model)  # type: ignore[arg-type]
            assert not result.valid
            error_messages = [error.message for error in result.errors]
            assert any("Empty OME metadata" in msg for msg in error_messages)

    def test_validate_nonexistent_dataset_path(self):
        """Test validation when dataset path doesn't exist in zarr store."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid zarr structure but remove a dataset directory
            test_path = Path(tmp_dir) / "test.zarr"
            test_path.mkdir()

            # Create .zgroup
            zgroup_path = test_path / ".zgroup"
            with zgroup_path.open("w") as f:
                json.dump({"zarr_format": 2}, f)

            # Create .zattrs referring to a dataset that won't exist
            zattrs_path = test_path / ".zattrs"
            attrs_data = {
                "multiscales": [
                    {
                        "version": "0.4",
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
                            },
                            {
                                "path": "nonexistent",  # This dataset won't exist
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2.0, 2.0]}
                                ],
                            },
                        ],
                    }
                ]
            }
            with zattrs_path.open("w") as f:
                json.dump(attrs_data, f)

            # Only create dataset "0", not "nonexistent"
            dataset_path = test_path / "0"
            dataset_path.mkdir()
            zarray_path = dataset_path / ".zarray"
            zarray_data = {
                "shape": [64, 64],
                "chunks": [32, 32],
                "dtype": "<u2",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "zarr_format": 2,
            }
            with zarray_path.open("w") as f:
                json.dump(zarray_data, f)

            model = v04.Image.model_validate(attrs_data)
            model.uri = str(zattrs_path)

            result = v04.validate_storage(model)
            assert not result.valid
            error_messages = [error.message for error in result.errors]
            assert any(
                "Dataset path does not exist: nonexistent" in msg
                for msg in error_messages
            )
