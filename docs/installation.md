---
icon: material/download
---

# Installing `yaozarrs`

The basic package (with no extras) supports metadata creation & validation,
but lacks structural validation and writing capabilities.

=== "from PyPI, with pip"

    ```sh
    pip install yaozarrs
    ```

=== "from PyPI, with uv"

    ```sh
    uv add yaozarrs
    ```

=== "from github"

    To install the bleeding-edge development version from GitHub,
    (shown here using the `io` extra as an example):

    ```sh
    pip install "yaozarrs[io] @ git+https://github.com/imaging-formats/yaozarrs.git"
    ```

=== "from conda"

    ```sh
    conda install conda-forge::yaozarrs
    ```

    Note that the conda distribution contains only the core package and no extras.
    You will need to manually install the following dependencies if you need them:
    
    - `conda-forge::fsspec` -  for `yaozarrs.open_group` and any structural validation.
    - `conda-forge::zarr` OR `conda-forge::tensorstore` - for writing support.

## Structural validation

To enable validation of Zarr hierarchies, include the `io` extra when installing.
This brings in [`fsspec`](https://github.com/fsspec/filesystem_spec) (but not zarr-python):

=== "with pip"

    ```sh
    pip install "yaozarrs[io]"
    ```

=== "with uv"

    ```sh
    uv add "yaozarrs[io]"
    ```

## Writing support

If you want to use the convenience functions in
[`yaozarrs.write`](./API_Reference/yaozarrs.write.v05.md), to create
complete OME-Zarr stores with array data, you will need to pick a zarr-array
library:

- ### `zarr-python`

    [`zarr-python`](https://zarr.readthedocs.io/en/stable/) is the reference
    implementation of the Zarr specification for Python.

    === "with pip"

        ```sh
        pip install "yaozarrs[write-zarr]"
        ```

    === "with uv"

        ```sh
        uv add "yaozarrs[write-zarr]"
        ```

    Then use `writer="zarr"` in the `yaozarrs.write` functions.

- ### `tensorstore`

    [Tensorstore](https://google.github.io/tensorstore/) is an
    alternative zarr-array library developed by Google, which can
    offer better I/O performance in most cases.

    === "with pip"

        ```sh
        pip install "yaozarrs[write-tensorstore]"
        ```

    === "with uv"

        ```sh
        uv add "yaozarrs[write-tensorstore]"
        ```

    Then use `writer="tensorstore"` in the `yaozarrs.write` functions.

- ### Custom backend

    If you don't want to use either of the provided backends, you can
    [implement your own array-writing functionality](./API_Reference/yaozarrs.write.v05.md#custom-writers)
