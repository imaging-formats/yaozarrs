/**
 * Interactive diagrams for the "Coordinate Systems & Transformations" page.
 *
 * - <ome-pixel-grid>: where pixel centers of two resolution levels land in
 *   physical space, with and without the half-pixel translation.
 * - <ome-transform-graph>: coordinate systems (nodes) and transformations
 *   (edges) of a few example hierarchies, with the metadata of each.
 */

import { LitElement, html, svg, css } from 'https://cdn.jsdelivr.net/npm/lit@3.1.0/+esm';

const sharedStyles = css`
  :host {
    display: block;
    margin: 1.5em 0;
    font-size: 0.7rem;
    --fg: var(--md-default-fg-color, #222);
    --fg-light: var(--md-default-fg-color--light, #666);
    --fg-lighter: var(--md-default-fg-color--lighter, #aaa);
    --fg-lightest: var(--md-default-fg-color--lightest, #eee);
    --bg: var(--md-default-bg-color, #fff);
    --code-bg: var(--md-code-bg-color, #f5f5f5);
    --primary: var(--md-primary-fg-color, #4051b5);
    --accent: var(--md-accent-fg-color, #526cfe);
  }
  .panel {
    border: 1px solid var(--fg-lightest);
    border-radius: 0.3rem;
    padding: 0.8rem;
    background: var(--bg);
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    align-items: center;
    margin-bottom: 0.6rem;
  }
  .seg {
    display: inline-flex;
    border: 1px solid var(--fg-lighter);
    border-radius: 0.25rem;
    overflow: hidden;
  }
  .seg button {
    font: inherit;
    border: none;
    background: transparent;
    color: var(--fg);
    padding: 0.25rem 0.6rem;
    cursor: pointer;
  }
  .seg button + button {
    border-left: 1px solid var(--fg-lighter);
  }
  .seg button.active {
    background: var(--primary);
    color: var(--md-primary-bg-color, #fff);
  }
  label {
    display: inline-flex;
    gap: 0.3rem;
    align-items: center;
    cursor: pointer;
  }
  svg {
    display: block;
    width: 100%;
    height: auto;
    user-select: none;
  }
  pre {
    margin: 0;
    padding: 0.6rem;
    background: var(--code-bg);
    border-radius: 0.2rem;
    overflow-x: auto;
    font-family: var(--md-code-font-family, monospace);
    font-size: 0.65rem;
    line-height: 1.4;
  }
  code {
    font-family: var(--md-code-font-family, monospace);
    background: var(--code-bg);
    padding: 0 0.2em;
    border-radius: 0.15rem;
  }
  .muted {
    color: var(--fg-light);
  }
`;

const FACTORS = [2, 4]; // the level 1 downsampling factors to choose from
const fmt = (v) => (Number.isInteger(v) ? v.toFixed(1) : String(+v.toFixed(3)));

// ---------------------------------------------------------------------------
// <ome-pixel-grid>
// ---------------------------------------------------------------------------

class OmePixelGrid extends LitElement {
  static properties = {
    factor: { type: Number },
    translate: { type: Boolean },
    hovered: { state: true },
  };

  static styles = [
    sharedStyles,
    css`
      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
        gap: 0.8rem;
        align-items: start;
      }
      @media (max-width: 700px) {
        .layout {
          grid-template-columns: minmax(0, 1fr);
        }
      }
      .readout {
        min-height: 4.8em;
        margin-bottom: 0.6rem;
      }
      .legend {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
      }
      .swatch {
        display: inline-block;
        width: 0.8em;
        height: 0.8em;
        margin-right: 0.3em;
        vertical-align: -0.1em;
        border-radius: 0.1em;
      }
      .px0 rect {
        fill: var(--fg-lightest);
        stroke: var(--fg-lighter);
        stroke-width: 0.6;
      }
      .px1 rect {
        fill: color-mix(in srgb, var(--primary) 12%, transparent);
        stroke: var(--primary);
        stroke-width: 1.6;
      }
      .px0 circle {
        fill: var(--fg-light);
      }
      .px1 circle {
        fill: var(--primary);
      }
      g.hover rect {
        fill: color-mix(in srgb, var(--accent) 30%, transparent);
        stroke: var(--accent);
      }
      g.hover circle {
        fill: var(--accent);
      }
      .axis line {
        stroke: var(--fg-lighter);
      }
      .axis text,
      .note {
        fill: var(--fg-light);
        font-size: 9px;
      }
      .origin {
        stroke: var(--fg);
        stroke-width: 1.2;
      }
    `,
  ];

  constructor() {
    super();
    this.factor = 2;
    this.translate = true;
    this.hovered = null;
    this.n0 = 8; // pixels per side at level 0
    this.s0 = 0.5; // pixel size at level 0 (micrometers)
  }

  _level(level) {
    const s = level === 0 ? this.s0 : this.s0 * this.factor;
    const n = level === 0 ? this.n0 : this.n0 / this.factor;
    const t = level === 0 || !this.translate ? 0 : (s - this.s0) / 2;
    return { s, n, t };
  }

  _transformJSON() {
    const { s, t } = this._level(1);
    const scale = { type: 'scale', scale: [s, s] };
    const tx = this.translate
      ? {
          type: 'sequence',
          input: { path: '1' },
          output: { name: 'physical' },
          transformations: [scale, { type: 'translation', translation: [t, t] }],
        }
      : { ...scale, input: { path: '1' }, output: { name: 'physical' } };
    return JSON.stringify(tx, null, 2).replace(/\[\s+([\d.]+),\s+([\d.]+)\s+\]/g, '[$1, $2]');
  }

  // the pixel of `level` that contains the physical point (y, x), if any
  _pixelAt(level, y, x) {
    const { s, n, t } = this._level(level);
    // (pixel i covers [i*s + t - s/2, i*s + t + s/2))
    const i = Math.floor((y - t) / s + 0.5);
    const j = Math.floor((x - t) / s + 0.5);
    return i >= 0 && j >= 0 && i < n && j < n ? [i, j] : null;
  }

  _onMove(e) {
    const svgEl = e.currentTarget;
    const pt = svgEl.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const { x, y } = pt.matrixTransform(svgEl.getScreenCTM().inverse());
    const [py, px] = [(y - this._ox) / this._ppu, (x - this._ox) / this._ppu];
    const hits = [0, 1].map((level) => this._pixelAt(level, py, px));
    this.hovered = hits.some(Boolean) ? hits : null;
  }

  _readout() {
    if (!this.hovered) {
      return html`<span class="muted">Hover over the pixels to see where their centers are.</span>`;
    }
    return this.hovered.map((hit, level) => {
      if (!hit) return html`<div>Level <b>${level}</b>: <span class="muted">(outside)</span></div>`;
      const { s, t } = this._level(level);
      const [cy, cx] = hit.map((k) => k * s + t);
      return html`<div>Level <b>${level}</b>, pixel <code>[${hit[0]}, ${hit[1]}]</code>:
        center at <code>(${fmt(cy)}, ${fmt(cx)})</code> µm, covering
        <code>[${fmt(cy - s / 2)}, ${fmt(cy + s / 2)})</code> ×
        <code>[${fmt(cx - s / 2)}, ${fmt(cx + s / 2)})</code></div>`;
    });
  }

  _pixels(level, ppu, ox) {
    const { s, n, t } = this._level(level);
    const hit = this.hovered && this.hovered[level];
    const out = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const cy = i * s + t;
        const cx = j * s + t;
        const hover = hit && hit[0] === i && hit[1] === j;
        out.push(svg`<g class=${hover ? 'hover' : ''}>
          <rect x=${ox + (cx - s / 2) * ppu} y=${ox + (cy - s / 2) * ppu}
            width=${s * ppu} height=${s * ppu} />
          <circle cx=${ox + cx * ppu} cy=${ox + cy * ppu} r=${level === 0 ? 1.6 : 3.2} />
        </g>`);
      }
    }
    return out;
  }

  render() {
    const ppu = 60; // svg units per micrometer
    const extent = this.n0 * this.s0; // 4 µm
    // the most that level 1 can stick out past the image (half of its biggest
    // pixel, when the translation is left out), so the axes go outside of that
    const overhang = ((this.s0 * Math.max(...FACTORS)) / 2) * ppu;
    const ax = 48; // where the axes are
    const ox = ax + overhang; // where the image starts
    const size = ox * 2 + extent * ppu;
    this._ppu = ppu;
    this._ox = ox;
    const ticks = [];
    for (let v = 0; v <= extent; v += 1) {
      ticks.push(svg`<g class="axis">
        <line x1=${ox + v * ppu} x2=${ox + v * ppu} y1=${ax - 26} y2=${ax - 20} />
        <text x=${ox + v * ppu} y=${ax - 29} text-anchor="middle">${v}</text>
        <line y1=${ox + v * ppu} y2=${ox + v * ppu} x1=${ax - 26} x2=${ax - 20} />
        <text y=${ox + v * ppu + 3} x=${ax - 29} text-anchor="end">${v}</text>
      </g>`);
    }
    return html`<div class="panel">
      <div class="controls">
        <span>Level 1 is downsampled by</span>
        <span class="seg">
          ${FACTORS.map(
            (f) => html`<button class=${this.factor === f ? 'active' : ''}
              @click=${() => (this.factor = f)}>×${f}</button>`,
          )}
        </span>
        <label>
          <input type="checkbox" .checked=${this.translate}
            @change=${(e) => (this.translate = e.target.checked)} />
          include the translation
        </label>
      </div>
      <div class="layout">
        <div>
          <svg viewBox="0 0 ${size} ${size}" role="img"
            aria-label="Pixel grids of two resolution levels in physical space"
            @mousemove=${this._onMove} @mouseleave=${() => (this.hovered = null)}>
            ${ticks}
            <text class="note" x=${ox + extent * ppu + 14} y=${ax - 29}>x (µm)</text>
            <text class="note" x=${ax - 29} y=${ox + extent * ppu + 20} text-anchor="middle">y (µm)</text>
            <g class="px0">${this._pixels(0, ppu, ox)}</g>
            <g class="px1">${this._pixels(1, ppu, ox)}</g>
            <g class="origin">
              <line x1=${ox - 6} x2=${ox + 6} y1=${ox} y2=${ox} />
              <line y1=${ox - 6} y2=${ox + 6} x1=${ox} x2=${ox} />
            </g>
          </svg>
          <div class="legend muted">
            <span><span class="swatch" style="background: var(--fg-lighter)"></span>level 0 (0.5 µm pixels)</span>
            <span><span class="swatch" style="background: var(--primary)"></span>level 1 (${fmt(this.s0 * this.factor)} µm pixels)</span>
            <span>+ origin <code>(0, 0)</code></span>
          </div>
        </div>
        <div>
          <div class="readout">${this._readout()}</div>
          <div class="muted" style="margin-bottom: 0.3rem">Level 1 dataset transformation:</div>
          <pre>${this._transformJSON()}</pre>
          <p class="muted">
            ${this.translate
              ? html`The big pixels exactly cover the small ones: the first big pixel's center
                  is halfway between the centers of the small pixels it covers.`
              : html`Without the translation, the first big pixel is centered on the first
                  small pixel's center, so the levels don't line up: the whole level 1 grid is
                  shifted up and to the left, sticking out past the top/left edges of the image,
                  and stopping short of the bottom/right ones.`}
          </p>
        </div>
      </div>
    </div>`;
  }
}

customElements.define('ome-pixel-grid', OmePixelGrid);

// ---------------------------------------------------------------------------
// <ome-transform-graph>
// ---------------------------------------------------------------------------

const YX = [
  { name: 'y', type: 'space', unit: 'micrometer' },
  { name: 'x', type: 'space', unit: 'micrometer' },
];
const ZYX = [{ name: 'z', type: 'space', unit: 'micrometer' }, ...YX];

const cs = (name, axes = YX) => ({ name, axes });

// the dataset transformation of level `i`, downsampling by 2 along the axes where
// `scale0` is below 1 (with the translation that keeps the levels aligned)
function levelTransform(i, scale0, output) {
  const scale = scale0.map((s) => (s < 1 ? s * 2 ** i : s));
  const ends = { input: { path: String(i) }, output: { name: output } };
  if (i === 0) return { type: 'scale', scale, ...ends };
  return {
    type: 'sequence', ...ends,
    transformations: [
      { type: 'scale', scale },
      { type: 'translation', translation: scale.map((s, k) => (s - scale0[k]) / 2) },
    ],
  };
}

// Each example: groups (one zarr.json each), nodes (arrays and coordinate
// systems, positioned on a 600 x H canvas), and edges (transformations).
const EXAMPLES = {
  image: {
    label: 'One image',
    height: 230,
    description: html`The simplest case: each resolution level (an array) maps into the
      image's intrinsic coordinate system, here called <code>physical</code>.`,
    groups: [{ id: 'img', label: 'image.zarr', x: 20, y: 20, w: 560, h: 190 }],
    nodes: [
      { id: 'a0', kind: 'array', label: '0', x: 120, y: 70, group: 'img', where: 'image.zarr/0 (array)' },
      { id: 'a1', kind: 'array', label: '1', x: 300, y: 70, group: 'img', where: 'image.zarr/1 (array)' },
      { id: 'a2', kind: 'array', label: '2', x: 480, y: 70, group: 'img', where: 'image.zarr/2 (array)' },
      {
        id: 'phys', kind: 'cs', label: 'physical', x: 300, y: 165, group: 'img', intrinsic: true,
        where: 'image.zarr/zarr.json → multiscales[0].coordinateSystems[0]', json: cs('physical'),
      },
    ],
    edges: [0, 1, 2].map((i) => ({
      from: `a${i}`, to: 'phys',
      where: `image.zarr/zarr.json → multiscales[0].datasets[${i}].coordinateTransformations[0]`,
      json: levelTransform(i, [0.5, 0.5], 'physical'),
    })),
  },

  extra: {
    label: 'Extra coordinate systems',
    height: 290,
    description: html`An image can have more coordinate systems, linked to the intrinsic one
      by the multiscale's own <code>coordinateTransformations</code>.  Here, light sheet
      data is stored as acquired, with a <code>deskewed</code> coordinate system for the
      corrected geometry.  Every transformation stored in the multiscale touches the
      intrinsic coordinate system.`,
    groups: [{ id: 'img', label: 'opm.zarr', x: 20, y: 20, w: 560, h: 250 }],
    nodes: [
      { id: 'a0', kind: 'array', label: '0', x: 200, y: 70, group: 'img', where: 'opm.zarr/0 (array)' },
      { id: 'a1', kind: 'array', label: '1', x: 400, y: 70, group: 'img', where: 'opm.zarr/1 (array)' },
      {
        id: 'intr', kind: 'cs', label: 'intrinsic', x: 300, y: 160, group: 'img', intrinsic: true,
        where: 'opm.zarr/zarr.json → multiscales[0].coordinateSystems[0]', json: cs('intrinsic', ZYX),
      },
      {
        id: 'desk', kind: 'cs', label: 'deskewed', x: 300, y: 235, group: 'img',
        where: 'opm.zarr/zarr.json → multiscales[0].coordinateSystems[1]', json: cs('deskewed', ZYX),
      },
    ],
    edges: [
      ...[0, 1].map((i) => ({
        from: `a${i}`, to: 'intr',
        where: `opm.zarr/zarr.json → multiscales[0].datasets[${i}].coordinateTransformations[0]`,
        json: levelTransform(i, [1.0, 0.4, 0.4], 'intrinsic'),
      })),
      {
        from: 'intr', to: 'desk',
        where: 'opm.zarr/zarr.json → multiscales[0].coordinateTransformations[0]',
        json: {
          type: 'affine', name: 'deskew', input: { name: 'intrinsic' }, output: { name: 'deskewed' },
          affine: [[0.5, 0, 0, 0], [0.866, 1, 0, 0], [0, 0, 1, 0]],
        },
      },
    ],
  },

  labels: {
    label: 'Image with labels',
    height: 300,
    description: html`The label image has its own intrinsic coordinate system.  Here, the
      labels were computed on a crop of the image, so the <em>image's</em> metadata links
      the two with a translation (referring to the label's coordinate system by
      <code>name</code> and <code>path</code>).  Without that transformation, they would
      usually be understood to be the same.`,
    groups: [
      { id: 'img', label: 'image.zarr', x: 20, y: 20, w: 560, h: 260 },
      { id: 'lbl', label: 'labels/cells', x: 320, y: 60, w: 240, h: 200 },
    ],
    nodes: [
      { id: 'a0', kind: 'array', label: '0', x: 110, y: 100, group: 'img', where: 'image.zarr/0 (array)' },
      { id: 'a1', kind: 'array', label: '1', x: 230, y: 100, group: 'img', where: 'image.zarr/1 (array)' },
      {
        id: 'phys', kind: 'cs', label: 'physical', x: 170, y: 215, group: 'img', intrinsic: true,
        where: 'image.zarr/zarr.json → multiscales[0].coordinateSystems[0]', json: cs('physical'),
      },
      { id: 'l0', kind: 'array', label: '0', x: 380, y: 120, group: 'lbl', where: 'image.zarr/labels/cells/0 (array)' },
      { id: 'l1', kind: 'array', label: '1', x: 500, y: 120, group: 'lbl', where: 'image.zarr/labels/cells/1 (array)' },
      {
        id: 'lphys', kind: 'cs', label: 'physical', x: 440, y: 215, group: 'lbl', intrinsic: true,
        where: 'image.zarr/labels/cells/zarr.json → multiscales[0].coordinateSystems[0]', json: cs('physical'),
      },
    ],
    edges: [
      ...['a0', 'a1'].map((id, i) => ({
        from: id, to: 'phys',
        where: `image.zarr/zarr.json → multiscales[0].datasets[${i}].coordinateTransformations[0]`,
        json: levelTransform(i, [0.5, 0.5], 'physical'),
      })),
      ...['l0', 'l1'].map((id, i) => ({
        from: id, to: 'lphys',
        where: `image.zarr/labels/cells/zarr.json → multiscales[0].datasets[${i}].coordinateTransformations[0]`,
        json: levelTransform(i, [0.5, 0.5], 'physical'),
      })),
      {
        from: 'lphys', to: 'phys',
        where: 'image.zarr/zarr.json → multiscales[0].coordinateTransformations[0]',
        json: {
          type: 'translation', translation: [100.0, 50.0],
          input: { name: 'physical', path: 'labels/cells' }, output: { name: 'physical' },
        },
      },
    ],
  },

  mosaic: {
    label: 'Mosaic (scene)',
    height: 320,
    description: html`A scene places two tiles in a shared <code>stage</code> coordinate system,
      defined by the scene itself.  Scene transformations refer to each tile's coordinate
      system by <code>name</code> and <code>path</code>, and to the scene's own by
      <code>name</code> only.`,
    groups: [
      { id: 'scene', label: 'mosaic.zarr', x: 20, y: 20, w: 560, h: 280 },
      { id: 't0', label: 'tile_0', x: 50, y: 60, w: 220, h: 135 },
      { id: 't1', label: 'tile_1', x: 330, y: 60, w: 220, h: 135 },
    ],
    nodes: [
      { id: 'a0', kind: 'array', label: '0', x: 160, y: 100, group: 't0', where: 'mosaic.zarr/tile_0/0 (array)' },
      {
        id: 'p0', kind: 'cs', label: 'physical', x: 160, y: 165, group: 't0', intrinsic: true,
        where: 'mosaic.zarr/tile_0/zarr.json → multiscales[0].coordinateSystems[0]', json: cs('physical'),
      },
      { id: 'a1', kind: 'array', label: '0', x: 440, y: 100, group: 't1', where: 'mosaic.zarr/tile_1/0 (array)' },
      {
        id: 'p1', kind: 'cs', label: 'physical', x: 440, y: 165, group: 't1', intrinsic: true,
        where: 'mosaic.zarr/tile_1/zarr.json → multiscales[0].coordinateSystems[0]', json: cs('physical'),
      },
      {
        id: 'stage', kind: 'cs', label: 'stage', x: 300, y: 262, group: 'scene',
        where: 'mosaic.zarr/zarr.json → scene.coordinateSystems[0]', json: cs('stage'),
      },
    ],
    edges: [
      ...[0, 1].map((i) => ({
        from: `a${i}`, to: `p${i}`,
        where: `mosaic.zarr/tile_${i}/zarr.json → multiscales[0].datasets[0].coordinateTransformations[0]`,
        json: levelTransform(0, [0.5, 0.5], 'physical'),
      })),
      ...[0, 1].map((i) => ({
        from: `p${i}`, to: 'stage',
        where: `mosaic.zarr/zarr.json → scene.coordinateTransformations[${i}]`,
        json: {
          type: 'translation', translation: [0.0, 270.0 * i],
          input: { path: `tile_${i}`, name: 'physical' }, output: { name: 'stage' },
        },
      })),
    ],
  },

  registration: {
    label: 'Registration (multi-hop)',
    height: 300,
    description: html`Three instruments image the same sample.  Instruments 1 and 3 are each
      registered onto instrument 2, directly between images (no scene coordinate
      system needed).  Instruments 1 and 3 aren't linked directly, but they are
      <em>connected</em>, through instrument 2: compose one transformation with the inverse
      of the other to get between them.`,
    groups: [
      { id: 'scene', label: 'sample.zarr', x: 20, y: 20, w: 560, h: 260 },
      { id: 'i1', label: 'instrument1', x: 40, y: 60, w: 160, h: 130 },
      { id: 'i2', label: 'instrument2', x: 220, y: 60, w: 160, h: 130 },
      { id: 'i3', label: 'instrument3', x: 400, y: 60, w: 160, h: 130 },
    ],
    nodes: [1, 2, 3].flatMap((n) => [
      { id: `a${n}`, kind: 'array', label: '0', x: 120 + 180 * (n - 1), y: 100, group: `i${n}`, where: `sample.zarr/instrument${n}/0 (array)` },
      {
        id: `p${n}`, kind: 'cs', label: 'physical', x: 120 + 180 * (n - 1), y: 160, group: `i${n}`, intrinsic: true,
        where: `sample.zarr/instrument${n}/zarr.json → multiscales[0].coordinateSystems[0]`, json: cs('physical', ZYX),
      },
    ]),
    edges: [
      ...[1, 2, 3].map((n) => ({
        from: `a${n}`, to: `p${n}`,
        where: `sample.zarr/instrument${n}/zarr.json → multiscales[0].datasets[0].coordinateTransformations[0]`,
        json: levelTransform(0, [[4, 2, 1][n - 1], [2, 1, 0.5][n - 1], [2, 1, 0.5][n - 1]], 'physical'),
      })),
      ...[1, 3].map((n, i) => ({
        from: `p${n}`, to: 'p2', curve: true,
        where: `sample.zarr/zarr.json → scene.coordinateTransformations[${i}]`,
        json: {
          type: 'affine',
          input: { path: `instrument${n}`, name: 'physical' },
          output: { path: 'instrument2', name: 'physical' },
          affine: n === 1
            ? [[1.0, 0.0, 0.0, 12.0], [0.0, 0.98, -0.17, 40.5], [0.0, 0.17, 0.98, -8.2]]
            : [[1.0, 0.0, 0.0, -3.0], [0.0, 1.0, 0.02, 101.7], [0.0, -0.02, 1.0, 64.0]],
        },
      })),
    ],
  },
};

const NODE = { array: { w: 44, h: 26 }, cs: { w: 96, h: 30 } };

// a short, readable rendering of the JSON (inner number lists on one line)
function prettyJSON(obj) {
  return JSON.stringify(obj, null, 2).replace(/\[\s+((?:-?[\d.]+,\s+)*-?[\d.]+)\s+\]/g, (_, inner) =>
    `[${inner.split(/,\s+/).join(', ')}]`,
  );
}

// point on the border of node `n`, in the direction of (tx, ty)
function borderPoint(n, tx, ty) {
  const { w, h } = NODE[n.kind];
  const dx = tx - n.x;
  const dy = ty - n.y;
  if (dx === 0 && dy === 0) return [n.x, n.y];
  const t = Math.min(w / 2 / Math.abs(dx || 1e-9), h / 2 / Math.abs(dy || 1e-9));
  return [n.x + dx * t, n.y + dy * t];
}

class OmeTransformGraph extends LitElement {
  static properties = {
    example: { type: String },
    selected: { state: true },
  };

  static styles = [
    sharedStyles,
    css`
      .group rect {
        fill: none;
        stroke: var(--fg-lighter);
        stroke-dasharray: 4 3;
      }
      .group text {
        fill: var(--fg-light);
        font-size: 10px;
        font-family: var(--md-code-font-family, monospace);
      }
      .node {
        cursor: pointer;
      }
      .node rect {
        stroke-width: 1.5;
      }
      .node.array rect {
        fill: var(--fg-lightest);
        stroke: var(--fg-lighter);
      }
      .node.cs rect {
        fill: color-mix(in srgb, var(--primary) 15%, var(--bg));
        stroke: var(--primary);
      }
      .node text {
        fill: var(--fg);
        font-size: 11px;
        font-family: var(--md-code-font-family, monospace);
        pointer-events: none;
      }
      .node.selected rect,
      .node:hover rect {
        stroke: var(--accent);
        stroke-width: 2.5;
      }
      .edge {
        cursor: pointer;
      }
      .edge path.line {
        fill: none;
        stroke: var(--fg-light);
        stroke-width: 1.5;
      }
      .edge path.hit {
        fill: none;
        stroke: transparent;
        stroke-width: 14;
      }
      .edge text {
        fill: var(--fg-light);
        font-size: 9.5px;
        paint-order: stroke;
        stroke: var(--bg);
        stroke-width: 4px;
        stroke-linejoin: round;
        pointer-events: none;
      }
      .edge.selected path.line,
      .edge:hover path.line {
        stroke: var(--accent);
        stroke-width: 2.5;
      }
      .edge.selected text,
      .edge:hover text {
        fill: var(--accent);
      }
      marker path {
        fill: var(--fg-light);
      }
      #arrow-hi path {
        fill: var(--accent);
      }
      .details h4 {
        margin: 0 0 0.3rem;
        font-size: 0.75rem;
      }
      .details p {
        margin: 0.3rem 0;
      }
      .description {
        margin: 0.2rem 0 0.6rem;
      }
    `,
  ];

  constructor() {
    super();
    this.example = 'mosaic';
    this.selected = null;
  }

  willUpdate(changed) {
    if (changed.has('example') || this.selected === null) {
      const ex = EXAMPLES[this.example];
      // select the most interesting edge by default: the last one
      this.selected = { kind: 'edge', index: ex.edges.length - 1 };
    }
  }

  _edgePath(ex, e) {
    const a = ex.nodes.find((n) => n.id === e.from);
    const b = ex.nodes.find((n) => n.id === e.to);
    if (e.curve) {
      // arc below the nodes, so it doesn't cross the groups' contents
      const [x1, y1] = [a.x, a.y + NODE[a.kind].h / 2];
      const [x2, y2] = [b.x + (a.x < b.x ? -20 : 20), b.y + NODE[b.kind].h / 2];
      const cy = Math.max(y1, y2) + 70;
      return { d: `M ${x1} ${y1} C ${x1} ${cy}, ${x2} ${cy}, ${x2} ${y2 + 2}`, mx: (x1 + x2) / 2, my: cy - 18 };
    }
    const [x1, y1] = borderPoint(a, b.x, b.y);
    const [x2, y2] = borderPoint(b, a.x, a.y);
    return { d: `M ${x1} ${y1} L ${x2} ${y2}`, mx: (x1 + x2) / 2, my: (y1 + y2) / 2 };
  }

  // a node's label, prefixed with its group's (unless that's the outermost group)
  _title(ex, n) {
    const g = ex.groups.find((g) => g.id === n.group);
    return g === ex.groups[0] ? n.label : `${g.label}/${n.label}`;
  }

  _details(ex) {
    const sel = this.selected;
    if (sel.kind === 'edge') {
      const e = ex.edges[sel.index];
      const a = ex.nodes.find((n) => n.id === e.from);
      const b = ex.nodes.find((n) => n.id === e.to);
      return html`<h4><code>${e.json.type}</code> transformation:
          <code>${this._title(ex, a)}</code> → <code>${this._title(ex, b)}</code></h4>
        <p class="muted">stored in <code>${e.where}</code></p>
        <pre>${prettyJSON(e.json)}</pre>`;
    }
    const n = ex.nodes.find((n) => n.id === sel.id);
    if (n.kind === 'array') {
      return html`<h4>Array <code>${n.label}</code></h4>
        <p class="muted"><code>${n.where}</code></p>
        <p>A resolution level of an image.  Its own (pixel) coordinates are mapped into the
          image's intrinsic coordinate system by its dataset transformation.</p>`;
    }
    return html`<h4>Coordinate system <code>${n.label}</code>
        ${n.intrinsic ? html`<span class="muted">(intrinsic)</span>` : ''}</h4>
      <p class="muted">defined in <code>${n.where}</code></p>
      <pre>${prettyJSON(n.json)}</pre>`;
  }

  render() {
    const ex = EXAMPLES[this.example];
    const sel = this.selected;
    return html`<div class="panel">
      <div class="controls">
        <span class="seg">
          ${Object.entries(EXAMPLES).map(
            ([key, e]) => html`<button class=${this.example === key ? 'active' : ''}
              @click=${() => (this.example = key)}>${e.label}</button>`,
          )}
        </span>
      </div>
      <p class="description">${ex.description}</p>
      <svg viewBox="0 0 600 ${ex.height}" role="img"
        aria-label="Graph of coordinate systems and transformations: ${ex.label}">
        <defs>
          ${['arrow', 'arrow-hi'].map(
            (id) => svg`<marker id=${id} viewBox="0 0 10 10" refX="9" refY="5"
              markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11"
              orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" /></marker>`,
          )}
        </defs>
        ${ex.groups.map(
          (g) => svg`<g class="group">
            <rect x=${g.x} y=${g.y} width=${g.w} height=${g.h} rx="6" />
            <text x=${g.x + 8} y=${g.y + 14}>${g.label}/</text>
          </g>`,
        )}
        ${ex.edges.map((e, i) => {
          const { d, mx, my } = this._edgePath(ex, e);
          const on = sel.kind === 'edge' && sel.index === i;
          return svg`<g class="edge ${on ? 'selected' : ''}"
              @click=${() => (this.selected = { kind: 'edge', index: i })}>
            <path class="hit" d=${d} />
            <path class="line" d=${d} marker-end="url(#${on ? 'arrow-hi' : 'arrow'})" />
            <text x=${mx + (e.curve ? 0 : 6)} y=${my + 3}
              text-anchor=${e.curve ? 'middle' : 'start'}>${e.json.type}</text>
          </g>`;
        })}
        ${ex.nodes.map((n) => {
          const { w, h } = NODE[n.kind];
          const on = sel.kind === 'node' && sel.id === n.id;
          return svg`<g class="node ${n.kind} ${on ? 'selected' : ''}"
              @click=${() => (this.selected = { kind: 'node', id: n.id })}>
            <rect x=${n.x - w / 2} y=${n.y - h / 2} width=${w} height=${h}
              rx=${n.kind === 'cs' ? h / 2 : 3} />
            <text x=${n.x} y=${n.y + 4} text-anchor="middle">${n.label}</text>
          </g>`;
        })}
      </svg>
      <div class="muted" style="margin: 0.2rem 0 0.6rem">
        Squares are arrays, rounded boxes are coordinate systems (dashed boxes are zarr groups).
        Click on anything to see its metadata.
      </div>
      <div class="details">${this._details(ex)}</div>
    </div>`;
  }
}

customElements.define('ome-transform-graph', OmeTransformGraph);
