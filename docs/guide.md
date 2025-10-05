# Yaozarrs guide to OME-Zarr

## What is OME-Zarr?

OME-Zarr is a file format for storing large, multi-dimensional bioimaging data.
It is based on the Zarr format, which is designed for the storage of chunked,
compressed, N-dimensional arrays. OME-Zarr extends Zarr by adding metadata
conventions specific to bioimaging, making it easier to store and share complex
imaging datasets.

!!! question "But what *is* it?"

    _To resolve a somewhat common confusion..._

    OME-Zarr is "just" [Zarr](https://zarr.dev) (the same zarr that other
    domains use). The "OME" part is a specification *on top of* the zarr
    format that additionally defines:

    1. **How domain specific metadata should be stored.**      
      The details are version-specific, but this generally defines the exact
      form of the data inside of the `.zattrs` or `zarr.json` files that
      accompany the zarr groups.

    1. **How datasets are organized.**  
      Beyond metadata, the OME-Zarr specification also defines how datasets
      should be organized. For example: it defines how the images collected
      across a multi-well plate experiment should be organized in a single
      Zarr directory, or how the different resolutions of a multi-scale
      (pyramidal) image should be stored.

## What is Zarr?

Zarr is a format and specification for the storage of chunked, compressed,
N-dimensional arrays.

Example

```tree
my_data.zarr/            # a root Zarr group.
    0                    # first array in the group (e.g. multiscale level 0)
        c/               # a chunk directory
            .../         # (chunk directories)
        zarr.json        # metadata for this array 0
    1                    # second array in the group (e.g. multiscale level 1)
        c/               # a chunk directory
            .../         # (chunk directories)
        zarr.json        # metadata for this array 1
    zarr.json            # metadata for the root group
```

## What can I express in OME-Zarr?

The primary data structures defined by the OME-Zarr specification are:

### Images

Multi-dimensional images (up to XYZCT), with support for multi-scale (pyramidal)
representations.

??? example "Example OME-Zarr image structure"

    === "v0.5"

        ```tree
        my_data.zarr/            # a root Zarr group.
            0                    # first array in the group (e.g. multiscale level 0)
                .../             # (chunk directories)
                zarr.json        # metadata for this array 0
            1                    # second array in the group (e.g. multiscale level 1)
                .../             # (chunk directories)
                zarr.json        # metadata for this array 1
            zarr.json            # contains {"attributes": {"ome": {"multiscales": [...]}}}
        ```

    === "v0.4"

        ```tree
        my_data.zarr/            # a root Zarr group.
            0                    # first array in the group (e.g. multiscale level 0)
                .../             # (chunk directories)
                .zarray          # marks this as a Zarr array
                .zattrs          # metadata for this array 0
            1                    # second array in the group (e.g. multiscale level 1)
                .../             # (chunk directories)
                .zarray          # marks this as a Zarr array
                .zattrs          # metadata for this array 1
            .zattrs              # metadata for the root group
            .zgroup              # marks this as a Zarr group
        ```

!!! question "But I don't have multi-scale data..."

    That's okay! You can store single-scale images in OME-Zarr format
    by simply including one array (e.g. `0/`).

### Plates

   Multi-well plate experiments, where each well can contain multiple images.

### Labels

   Segmentation masks or label images associated with the primary image data.
