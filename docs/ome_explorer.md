---
icon: lucide/sparkles
title: Interactive OME-Zarr Explorer
hide:
    - navigation
    - toc
---

# :lucide-sparkles: Interactive OME-Zarr Explorer

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

## Understanding the Parameters

<div class="grid cards" markdown>

- :lucide-tag:{ .lg .middle } **Name**

    ---
    Axis identifier (e.g., `x`, `y`, `z`, `t`, `c`)

    Common conventions: x/y/z for spatial, t for time, c for channel

- :lucide-layers:{ .lg .middle } **Type**

    ---
    Axis semantic type from the OME-NGFF spec

    Options: `space`, `time`, `channel`

- :lucide-ruler:{ .lg .middle } **Unit**

    ---
    Physical unit for the axis (optional)

    Examples: `micrometer`, `second`, `nanometer`

- :lucide-scaling:{ .lg .middle } **Scale**

    ---
    Physical spacing per pixel at level 0

    Example: 0.5 = 0.5 micrometers per pixel

- :lucide-move:{ .lg .middle } **Translation**

    ---
    Origin offset in physical coordinates

    Used for positioning images with stage coordinates

- :lucide-pyramid:{ .lg .middle } **Scale Factor**

    ---
    Downsampling factor per pyramid level

    Typically 2 for spatial axes, 1 for others

</div>

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

## Next Steps

- Check out the [User Guide](user_guide.md) for comprehensive documentation
- Explore the [API Reference](api/yaozarrs.md) for all available models
