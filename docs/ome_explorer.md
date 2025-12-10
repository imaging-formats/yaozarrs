---
icon: lucide/sparkles
title: Interactive OME-Zarr Explorer
hide:
    - navigation
    - toc
---

<script type="module" src="/javascripts/ome_explorer.js"></script>

!!! tip "What is this?"
    An **interactive educational tool** to understand how OME-NGFF metadata is constructed.
    Define your image dimensions and watch the spec JSON and yaozarrs code generate in real-time!

## How to Use

1. **Choose a preset** or build custom dimensions from scratch
2. **Edit the dimension table** - change names, types, units, scales
3. **Toggle between v0.4 and v0.5** to see version differences
4. **View the output** as Spec JSON or yaozarrs Python code
5. **Copy to clipboard** to use in your projects

<ome-explorer></ome-explorer>

## Tips

!!! tip "Scale Factor Magic"
    By default, spatial dimensions (x/y/z) downsample by 2× at each pyramid level.
    Non-spatial dimensions (t/c) don't downsample. Change **Scale Factor** to override!

!!! warning "Translation Usage"
    Only add translations if you have actual stage positions or need to position images
    in physical space. Most single images don't need this.

!!! info "DimSpec Convenience"
    The yaozarrs `DimSpec` class shown in the Python output is a convenience wrapper.
    It automatically generates axes and coordinate transformations from simpler inputs!
