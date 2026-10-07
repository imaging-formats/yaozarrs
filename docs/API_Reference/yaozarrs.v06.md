# <code class="doc-symbol-module"></code> `yaozarrs.v06`

Specification (in development): the `main` branch of
<https://github.com/ome/ngff-spec> **is** the v0.6 (`0.6.dev4`) spec.

!!! warning "In-development version"
    v0.6 is still a development version. These models accept/emit the exact
    version string `"0.6.dev4"`. The headline change from v0.5 is the
    *coordinate systems + coordinate transformations* redesign (RFC-5): the
    multiscale `axes` field is **replaced** by named `coordinateSystems`, and
    dataset transforms now carry `input`/`output`.

!!! info "What changed from v0.5? (quick reference)"
    | Area | v0.5 | v0.6 |
    | ---- | ---- | ---- |
    | version string | `"0.5"` | `"0.6.dev4"` |
    | multiscale axes | `multiscales[].axes` | `multiscales[].coordinateSystems[].axes` |
    | dataset transform | bare `scale`/`translation` | `scale`/`identity`/`sequence` + `input`/`output` |
    | transform types | `scale`, `translation` | + `identity`, `mapAxis`, `affine`, `rotation`, `bijection`, `sequence`, `byDimension`, `displacements`, `coordinates` |
    | axis `type` | required (enum) | optional, free-form (+ `longName`, `discrete`) |
    | `scale` values | any number | strictly `> 0` |
    | `image-label` | optional on a label | **required** on a label; inner `version` removed |
    | new top-level object | — | `scene` (cross-image transforms) |

## Coordinate Systems

A coordinate system is a named set of axes. In v0.6, `axes` are nested inside a
coordinate system rather than living directly on the multiscale.

::: yaozarrs.v06._coordinate_systems

## Axes

::: yaozarrs.v06._axes
      options:
        members:
          - SpaceAxis
          - TimeAxis
          - ChannelAxis
          - CustomAxis

## Coordinate Transformations

The v0.6 transform "zoo". Datasets only use `scale`/`identity`/`sequence`; the
rest appear in multiscale-level `coordinateTransformations` and in a `scene`.

::: yaozarrs.v06._transforms

## Images

::: yaozarrs.v06._image

## Labels

::: yaozarrs.v06._labels

## Plates

::: yaozarrs.v06._plate

## Collections

::: yaozarrs.v06._bf2raw

## Scenes

!!! warning "Experimental"
    `scene` is a new, still-changing v0.6 object (e.g. `arrayCoordinateSystem` is
    slated for removal). Modeled for completeness.

::: yaozarrs.v06._scene
