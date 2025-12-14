---
icon: material/auto-fix
title: OME-Zarr Explorer
hide:
    - toc
instanthtml: false
---



!!! question "What is this?"
    This educational app demonstrates how OME-NGFF zarr hierarchies are
    structured with associated metadata.

    Define your image dimensions and view the generated hierarchy and metadata in real-time.

    The file tree on the left demonstrates what the OME-Zarr hierarchy would look like when saved to disk,
    the JSON viewer on the right shows the metadata associated with each group or array, and the 
    Python tab shows how you would create the metadata programmatically using the `yaozarrs` library.

    **This is meant for educational purposes only and is not intended for production use.**

<script type="module" src="/yaozarrs/javascripts/ome_explorer.js"></script>
<ome-explorer></ome-explorer>
