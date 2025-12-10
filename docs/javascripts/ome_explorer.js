/**
 * Interactive OME-Zarr Metadata Explorer
 * Educational tool for understanding OME-NGFF metadata structure
 */

import { LitElement, html, css } from 'https://cdn.jsdelivr.net/npm/lit@3.1.0/+esm';
import { unsafeHTML } from 'https://cdn.jsdelivr.net/npm/lit@3.1.0/directives/unsafe-html.js/+esm';

class OmeExplorer extends LitElement {
  static properties = {
    dimensions: { type: Array },
    version: { type: String },
    mode: { type: String },
    activeTab: { type: String },
    numLevels: { type: Number },
    copyButtonText: { type: String },
    expandedNodes: { type: Object },
    selectedNode: { type: String },
    validationErrors: { type: Array },
  };

  static styles = css`
    :host {
      display: block;
      font-family: var(--md-text-font-family, 'Inter', -apple-system, BlinkMacSystemFont, sans-serif);
      --primary-color: var(--md-primary-fg-color, #4051b5);
      --primary-color-light: #5c6bc0;
      --accent-color: var(--md-accent-fg-color, #526cfe);
      --bg-color: var(--md-default-bg-color, #fff);
      --code-bg: #1e1e2e;
      --code-bg-lighter: #262637;
      --border-color: rgba(0, 0, 0, 0.08);
      --border-color-strong: rgba(0, 0, 0, 0.15);
      --text-muted: #64748b;
      --success-color: #10b981;
      --warning-color: #f59e0b;
      
      /* Syntax highlighting - Catppuccin Mocha inspired */
      --syn-keyword: #cba6f7;
      --syn-string: #a6e3a1;
      --syn-number: #fab387;
      --syn-comment: #6c7086;
      --syn-function: #89b4fa;
      --syn-property: #89dceb;
      --syn-punctuation: #9399b2;
      --syn-operator: #94e2d5;
      --syn-builtin: #f38ba8;
      --syn-constant: #fab387;
    }

    * {
      box-sizing: border-box;
    }

    .explorer-container {
      border: 1px solid var(--border-color-strong);
      border-radius: 8px;
      overflow: hidden;
      background: var(--bg-color);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }

    .toolbar {
      display: flex;
      gap: 0.75rem;
      padding: 0.5rem 0.75rem;
      background: linear-gradient(135deg, 
        rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.03) 0%,
        rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.06) 100%);
      border-bottom: 1px solid var(--border-color);
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-start;
    }

    .toolbar-group {
      display: flex;
      gap: 0.375rem;
      align-items: center;
    }

    .toolbar-group label {
      font-size: 0.625rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .toolbar-separator {
      width: 1px;
      height: 20px;
      background: var(--border-color-strong);
      margin: 0 0.25rem;
    }

    button, select, input[type="number"], input[type="text"] {
      padding: 0.3rem 0.5rem;
      border: 1px solid var(--border-color-strong);
      border-radius: 4px;
      background: white;
      cursor: pointer;
      font-size: 0.75rem;
      transition: all 0.15s ease;
      font-family: inherit;
      color: inherit;
    }

    button:hover {
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.08);
      border-color: var(--primary-color-light);
    }

    button:active {
      transform: translateY(0);
    }

    button.primary {
      background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-color-light) 100%);
      color: white;
      border-color: transparent;
      font-weight: 500;
      box-shadow: 0 1px 2px rgba(64, 81, 181, 0.2);
    }

    button.primary:hover {
      background: linear-gradient(135deg, var(--primary-color-light) 0%, var(--accent-color) 100%);
    }

    .preset-btn {
      font-weight: 500;
    }

    .levels-input {
      width: 48px !important;
      text-align: center;
      font-weight: 500;
      padding: 0.3rem 0.25rem !important;
    }

    .main-content {
      display: flex;
      flex-direction: column;
    }

    .input-panel {
      border-bottom: 1px solid var(--border-color);
      padding: 0.625rem 0.75rem;
      background: var(--bg-color);
    }

    .output-panel {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }

    .section-header {
      display: none;
    }

    .dimension-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.75rem;
    }

    .dimension-table th {
      text-align: left;
      padding: 0.375rem 0.5rem;
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.04);
      font-weight: 600;
      font-size: 0.625rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-color-strong);
      white-space: nowrap;
    }

    .dimension-table th:first-child {
      border-radius: 4px 0 0 0;
    }

    .dimension-table th:last-child {
      border-radius: 0 4px 0 0;
    }

    /* Column widths - compact distribution */
    .dimension-table th:nth-child(1),
    .dimension-table td:nth-child(1) { width: 24px; }  /* Drag handle */
    .dimension-table th:nth-child(2),
    .dimension-table td:nth-child(2) { width: 50px; }  /* Name */
    .dimension-table th:nth-child(3),
    .dimension-table td:nth-child(3) { width: 80px; }  /* Type */
    .dimension-table th:nth-child(4),
    .dimension-table td:nth-child(4) { width: 90px; }  /* Unit */
    .dimension-table th:nth-child(5),
    .dimension-table td:nth-child(5) { width: 60px; }  /* Scale */
    .dimension-table th:nth-child(6),
    .dimension-table td:nth-child(6) { width: 60px; }  /* Trans */
    .dimension-table th:nth-child(7),
    .dimension-table td:nth-child(7) { width: 60px; }  /* Factor */
    .dimension-table th:nth-child(8),
    .dimension-table td:nth-child(8) { width: 32px; }  /* Delete */

    .dimension-table td {
      padding: 0.25rem 0.375rem;
      border-bottom: 1px solid var(--border-color);
      vertical-align: middle;
    }

    .dimension-table tbody tr:last-child td {
      border-bottom: none;
    }

    .dimension-table tbody tr {
      cursor: move;
      cursor: grab;
    }

    .dimension-table tbody tr:active {
      cursor: grabbing;
    }

    .dimension-table tbody tr.dragging {
      opacity: 0.3;
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.05);
    }

    .dimension-table tbody tr.drag-placeholder-top {
      box-shadow: inset 0 3px 0 0 var(--primary-color);
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.04);
      animation: pulse-shadow-top 0.8s ease-in-out infinite;
    }

    .dimension-table tbody tr.drag-placeholder-bottom {
      box-shadow: inset 0 -3px 0 0 var(--primary-color);
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.04);
      animation: pulse-shadow-bottom 0.8s ease-in-out infinite;
    }

    @keyframes pulse-shadow-top {
      0%, 100% {
        box-shadow: inset 0 3px 0 0 var(--primary-color);
      }
      50% {
        box-shadow: inset 0 4px 0 0 var(--primary-color),
                    0 -2px 12px rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.4);
      }
    }

    @keyframes pulse-shadow-bottom {
      0%, 100% {
        box-shadow: inset 0 -3px 0 0 var(--primary-color);
      }
      50% {
        box-shadow: inset 0 -4px 0 0 var(--primary-color),
                    0 2px 12px rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.4);
      }
    }

    .dimension-table tr:hover td {
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.02);
    }

    .drag-handle {
      cursor: grab;
      color: var(--text-muted);
      font-size: 0.75rem;
      padding: 0 0.25rem;
      opacity: 0.5;
      transition: opacity 0.2s;
    }

    tr:hover .drag-handle {
      opacity: 1;
    }

    .drag-handle:active {
      cursor: grabbing;
    }

    .dimension-table input,
    .dimension-table select {
      width: 100%;
      padding: 0.25rem 0.375rem;
      font-size: 0.75rem;
      border-radius: 3px;
      border: 1px solid var(--border-color);
      background: white;
      transition: all 0.15s ease;
    }

    .dimension-table input:focus,
    .dimension-table select:focus {
      outline: none;
      border-color: var(--primary-color-light);
      box-shadow: 0 0 0 1px rgba(64, 81, 181, 0.15);
    }

    .dimension-table input[type="number"] {
      text-align: right;
      padding-right: 0.125rem;
      -moz-appearance: textfield;
    }

    .dimension-table input[type="number"]::-webkit-outer-spin-button,
    .dimension-table input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }

    .delete-btn {
      padding: 0.125rem 0.25rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      border: none;
      background: transparent;
      opacity: 0.5;
      transition: all 0.15s;
    }

    .delete-btn:hover {
      color: #ef4444;
      background: rgba(239, 68, 68, 0.1);
      opacity: 1;
    }

    .add-dimension {
      margin-top: 0.5rem;
      width: 100%;
      padding: 0.375rem 0.5rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      border-radius: 4px;
    }

    .tab-bar {
      display: flex;
      background: var(--code-bg);
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding: 0.375rem 0.625rem 0 0.625rem;
      gap: 0.125rem;
      align-items: flex-end;
    }

    .tab {
      padding: 0.375rem 0.75rem;
      border: none;
      background: transparent;
      cursor: pointer;
      font-weight: 500;
      font-size: 0.6875rem;
      position: relative;
      border-radius: 4px 4px 0 0;
      color: rgba(255, 255, 255, 0.5);
      transition: all 0.15s;
    }

    .tab:hover {
      color: rgba(255, 255, 255, 0.8);
      background: rgba(255, 255, 255, 0.05);
    }

    .tab.active {
      background: var(--code-bg-lighter);
      color: white;
    }

    .tab-spacer {
      flex: 1;
    }

    .version-toggle {
      display: inline-flex;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(255, 255, 255, 0.05);
      margin-bottom: 0.375rem;
    }

    .version-toggle button {
      border: none;
      border-radius: 0;
      padding: 0.25rem 0.5rem;
      font-weight: 500;
      font-size: 0.625rem;
      background: transparent;
      position: relative;
      color: rgba(255, 255, 255, 0.6);
    }

    .version-toggle button:not(:last-child)::after {
      content: '';
      position: absolute;
      right: 0;
      top: 20%;
      height: 60%;
      width: 1px;
      background: rgba(255, 255, 255, 0.15);
    }

    .version-toggle button:hover {
      color: rgba(255, 255, 255, 0.9);
      background: rgba(255, 255, 255, 0.08);
    }

    .version-toggle button.active {
      background: var(--primary-color);
      color: white;
    }

    .version-toggle button.active::after {
      display: none;
    }

    .tab-content {
      flex: 1;
      position: relative;
      background: var(--code-bg-lighter);
      display: flex;
      flex-direction: column;
      min-height: 280px;
    }

    .code-block {
      background: transparent;
      color: #cdd6f4;
      padding: 0.75rem;
      margin: 0;
      overflow: auto;
      flex: 1;
      font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
      font-size: 0.6875rem;
      line-height: 1.5;
      white-space: pre;
      min-height: 0;
    }

    .copy-button {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      padding: 0.25rem 0.5rem;
      font-size: 0.625rem;
      z-index: 10;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: rgba(255, 255, 255, 0.7);
      border-radius: 3px;
      display: flex;
      align-items: center;
      gap: 0.25rem;
      backdrop-filter: blur(8px);
    }

    .copy-button:hover {
      background: rgba(255, 255, 255, 0.15);
      color: white;
    }

    .copy-button.copied {
      background: rgba(16, 185, 129, 0.2);
      border-color: rgba(16, 185, 129, 0.3);
      color: #10b981;
    }

    /* Syntax highlighting */
    .syn-keyword { color: var(--syn-keyword); }
    .syn-string { color: var(--syn-string); }
    .syn-number { color: var(--syn-number); }
    .syn-comment { color: var(--syn-comment); font-style: italic; }
    .syn-function { color: var(--syn-function); }
    .syn-property { color: var(--syn-property); }
    .syn-punctuation { color: var(--syn-punctuation); }
    .syn-operator { color: var(--syn-operator); }
    .syn-builtin { color: var(--syn-builtin); }
    .syn-constant { color: var(--syn-constant); }
    .syn-class { color: #f9e2af; }
    .syn-decorator { color: #f38ba8; }

    .settings-group {
      border: none;
      padding: 0;
      margin: 0;
    }

    /* Tree View Styles */
    .code-area {
      display: flex;
      flex-direction: row;
    }

    .tree-panel {
      display: flex;
      flex-direction: column;
      background: var(--code-bg);
      border-right: 1px solid rgba(255, 255, 255, 0.1);
      flex: 0 0 280px;
      min-width: 200px;
    }

    .tree-view {
      flex: 1;
      background: var(--code-bg);
      padding: 0.5rem;
      font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
      font-size: 0.6875rem;
      overflow: auto;
      min-height: 0;
    }

    .tree-node {
      user-select: none;
    }

    .tree-item {
      display: flex;
      align-items: center;
      padding: 0.125rem 0.25rem;
      cursor: pointer;
      border-radius: 3px;
      color: #cdd6f4;
      transition: background 0.1s;
    }

    .tree-item:hover {
      background: rgba(255, 255, 255, 0.08);
    }

    .tree-item.selected {
      background: rgba(137, 180, 250, 0.2);
      color: #89b4fa;
    }

    .tree-toggle {
      width: 14px;
      height: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #6c7086;
      font-size: 0.5rem;
      flex-shrink: 0;
    }

    .tree-toggle.expandable {
      cursor: pointer;
    }

    .tree-toggle.expandable:hover {
      color: #cdd6f4;
    }

    .tree-icon {
      width: 14px;
      height: 14px;
      margin-right: 0.25rem;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.625rem;
    }

    .tree-icon.folder { color: #fab387; }
    .tree-icon.file { color: #89b4fa; }
    .tree-icon.chunk { color: #6c7086; }

    .tree-label {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .tree-children {
      padding-left: 1rem;
    }

    .tree-children.collapsed {
      display: none;
    }

    .tree-info-panel {
      padding: 0.5rem 0.75rem;
      background: rgba(0, 0, 0, 0.3);
      color: #a6adc8;
      font-size: 0.625rem;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      min-height: 44px;
    }

    .code-output {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }

    .tree-info-title {
      color: #cdd6f4;
      font-weight: 600;
      font-size: 0.6875rem;
      margin-bottom: 0.125rem;
    }

    .tree-info-desc {
      color: #a6adc8;
    }

    .tree-info-hint {
      color: #6c7086;
      font-style: italic;
      margin-top: auto;
    }

    /* Validation Warning Panel */
    .validation-panel {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(239, 68, 68, 0.1) 100%);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 4px;
      padding: 0.625rem 0.75rem;
      margin-top: 0.75rem;
      font-size: 0.6875rem;
    }

    .validation-panel.error {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.1) 100%);
      border-color: rgba(239, 68, 68, 0.3);
    }

    .validation-title {
      font-weight: 600;
      margin-bottom: 0.375rem;
      display: flex;
      align-items: center;
      gap: 0.375rem;
      color: #f59e0b;
    }

    .validation-panel.error .validation-title {
      color: #ef4444;
    }

    .validation-icon {
      font-size: 0.875rem;
    }

    .validation-list {
      margin: 0;
      padding-left: 1.25rem;
      line-height: 1.6;
    }

    .validation-list li {
      margin-bottom: 0.25rem;
      color: #78716c;
    }

    .validation-list li strong {
      color: #57534e;
    }

    .validation-hint {
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid rgba(0, 0, 0, 0.08);
      font-style: italic;
      color: #78716c;
    }
  `;

  constructor() {
    super();
    this.version = 'v0.5';
    this.mode = 'image';
    this.activeTab = 'json';
    this.numLevels = 3;
    this.copyButtonText = 'Copy';
    this.expandedNodes = { root: true, '0': false, '1': false, '2': false };
    this.selectedNode = 'root-meta';
    this.validationErrors = [];
    this.draggedIndex = null;
    this.dimensions = [
      { name: 'c', type: 'channel', unit: '', scale: 1, translation: 0, scaleFactor: 1 },
      { name: 'z', type: 'space', unit: 'micrometer', scale: 2, translation: 0, scaleFactor: 2 },
      { name: 'y', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
      { name: 'x', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
    ];
    this.validateDimensions();
  }

  // Valid units from yaozarrs._axis
  getValidUnits(type) {
    const spaceUnits = [
      'angstrom', 'attometer', 'centimeter', 'decimeter', 'exameter',
      'femtometer', 'foot', 'gigameter', 'hectometer', 'inch',
      'kilometer', 'megameter', 'meter', 'micrometer', 'mile',
      'millimeter', 'nanometer', 'parsec', 'petameter', 'picometer',
      'terameter', 'yard', 'yoctometer', 'yottameter', 'zeptometer', 'zettameter'
    ];

    const timeUnits = [
      'attosecond', 'centisecond', 'day', 'decisecond', 'exasecond',
      'femtosecond', 'gigasecond', 'hectosecond', 'hour', 'kilosecond',
      'megasecond', 'microsecond', 'millisecond', 'minute', 'nanosecond',
      'petasecond', 'picosecond', 'second', 'terasecond', 'yoctosecond',
      'yottasecond', 'zeptosecond', 'zettasecond'
    ];

    if (type === 'space') return spaceUnits;
    if (type === 'time') return timeUnits;
    return []; // channel can have any unit
  }

  // Validate dimensions according to OME-NGFF spec
  validateDimensions() {
    const errors = [];
    const warnings = [];

    // Count axes by type
    const typeCounts = {
      space: this.dimensions.filter(d => d.type === 'space').length,
      time: this.dimensions.filter(d => d.type === 'time').length,
      channel: this.dimensions.filter(d => d.type === 'channel').length,
    };

    // Rule 1: Must have 2-5 dimensions total
    if (this.dimensions.length < 2) {
      errors.push({
        type: 'error',
        message: `Too few dimensions (${this.dimensions.length}). OME-NGFF requires at least 2 dimensions.`,
        hint: 'Add more spatial dimensions (x, y, or z).'
      });
    }
    if (this.dimensions.length > 5) {
      errors.push({
        type: 'error',
        message: `Too many dimensions (${this.dimensions.length}). OME-NGFF allows maximum 5 dimensions.`,
        hint: 'Remove some dimensions to comply with the spec.'
      });
    }

    // Rule 2: Must have 2 or 3 space axes
    if (typeCounts.space < 2) {
      errors.push({
        type: 'error',
        message: `Too few spatial dimensions (${typeCounts.space}). Must have 2-3 space axes.`,
        hint: 'Add spatial dimensions (x, y) or (x, y, z).'
      });
    }
    if (typeCounts.space > 3) {
      errors.push({
        type: 'error',
        message: `Too many spatial dimensions (${typeCounts.space}). Maximum is 3 space axes.`,
        hint: 'Biological images are at most 3D (x, y, z).'
      });
    }

    // Rule 3: At most one time axis
    if (typeCounts.time > 1) {
      errors.push({
        type: 'error',
        message: `Too many time dimensions (${typeCounts.time}). At most 1 time axis allowed.`,
        hint: 'Remove duplicate time dimensions.'
      });
    }

    // Rule 4: At most one channel axis
    if (typeCounts.channel > 1) {
      errors.push({
        type: 'error',
        message: `Too many channel dimensions (${typeCounts.channel}). At most 1 channel axis allowed.`,
        hint: 'Merge channels into a single channel dimension.'
      });
    }

    // Rule 5: Check ordering - must be [time,] [channel,] space...
    const typeOrder = { time: 0, channel: 1, space: 2 };
    const actualOrder = this.dimensions.map(d => typeOrder[d.type] ?? 3);
    const sortedOrder = [...actualOrder].sort((a, b) => a - b);

    if (JSON.stringify(actualOrder) !== JSON.stringify(sortedOrder)) {
      errors.push({
        type: 'error',
        message: 'Dimensions are not in the required order.',
        hint: 'Order must be: [time,] [channel,] then space axes. Try: t, c, z, y, x'
      });
    }

    // Rule 6: Check for duplicate names
    const names = this.dimensions.map(d => d.name);
    const duplicates = names.filter((name, i) => names.indexOf(name) !== i);
    if (duplicates.length > 0) {
      errors.push({
        type: 'error',
        message: `Duplicate axis names: ${[...new Set(duplicates)].join(', ')}`,
        hint: 'Each dimension must have a unique name.'
      });
    }

    // Rule 7: Validate units (warnings, not errors)
    this.dimensions.forEach((dim, i) => {
      if (dim.unit) {
        const validUnits = this.getValidUnits(dim.type);
        if (validUnits.length > 0 && !validUnits.includes(dim.unit)) {
          warnings.push({
            type: 'warning',
            message: `Dimension "${dim.name}": unit "${dim.unit}" is not a recognized ${dim.type} unit.`,
            hint: `Valid ${dim.type} units include: ${validUnits.slice(0, 5).join(', ')}, ...`
          });
        }
      }
    });

    this.validationErrors = [...errors, ...warnings];
  }

  // Syntax highlighting for JSON
  highlightJSON(json) {
    return json
      // Strings (property values)
      .replace(/"([^"\\]|\\.)*"/g, (match) => {
        // Check if it's a property name (followed by :)
        return `<span class="syn-string">${this.escapeHtml(match)}</span>`;
      })
      // Numbers
      .replace(/\b(-?\d+\.?\d*)\b/g, '<span class="syn-number">$1</span>')
      // Booleans and null
      .replace(/\b(true|false|null)\b/g, '<span class="syn-constant">$1</span>')
      // Property names (already in strings, so we need to identify them by context)
      .replace(/<span class="syn-string">"([^"]+)"<\/span>\s*:/g, 
        '<span class="syn-property">"$1"</span>:');
  }

  // Syntax highlighting for Python
  highlightPython(code) {
    // We'll tokenize and rebuild to avoid regex issues with HTML escaping
    const lines = code.split('\n');
    const highlightedLines = lines.map(line => {
      // Check if this line is a comment
      const commentMatch = line.match(/^(\s*)(#.*)$/);
      if (commentMatch) {
        return this.escapeHtml(commentMatch[1]) + 
          '<span class="syn-comment">' + this.escapeHtml(commentMatch[2]) + '</span>';
      }
      
      // Check for inline comment
      const inlineCommentMatch = line.match(/^(.*?)(#.*)$/);
      let mainPart = line;
      let commentPart = '';
      if (inlineCommentMatch && !this.isInsideString(inlineCommentMatch[1], inlineCommentMatch[1].length)) {
        mainPart = inlineCommentMatch[1];
        commentPart = '<span class="syn-comment">' + this.escapeHtml(inlineCommentMatch[2]) + '</span>';
      }
      
      return this.highlightPythonLine(mainPart) + commentPart;
    });
    
    return highlightedLines.join('\n');
  }
  
  isInsideString(text, pos) {
    let inSingle = false;
    let inDouble = false;
    for (let i = 0; i < pos; i++) {
      if (text[i] === '"' && !inSingle) inDouble = !inDouble;
      if (text[i] === "'" && !inDouble) inSingle = !inSingle;
    }
    return inSingle || inDouble;
  }
  
  highlightPythonLine(line) {
    // Simple tokenization approach
    const tokens = [];
    let i = 0;
    
    while (i < line.length) {
      // Skip whitespace
      if (/\s/.test(line[i])) {
        let ws = '';
        while (i < line.length && /\s/.test(line[i])) {
          ws += line[i++];
        }
        tokens.push({ type: 'ws', value: ws });
        continue;
      }
      
      // String (double quote)
      if (line[i] === '"') {
        let str = '"';
        i++;
        while (i < line.length && line[i] !== '"') {
          if (line[i] === '\\' && i + 1 < line.length) {
            str += line[i++];
          }
          str += line[i++];
        }
        if (i < line.length) str += line[i++];
        tokens.push({ type: 'string', value: str });
        continue;
      }
      
      // String (single quote)
      if (line[i] === "'") {
        let str = "'";
        i++;
        while (i < line.length && line[i] !== "'") {
          if (line[i] === '\\' && i + 1 < line.length) {
            str += line[i++];
          }
          str += line[i++];
        }
        if (i < line.length) str += line[i++];
        tokens.push({ type: 'string', value: str });
        continue;
      }
      
      // Number
      if (/\d/.test(line[i]) || (line[i] === '.' && i + 1 < line.length && /\d/.test(line[i + 1]))) {
        let num = '';
        while (i < line.length && /[\d.]/.test(line[i])) {
          num += line[i++];
        }
        tokens.push({ type: 'number', value: num });
        continue;
      }
      
      // Word (identifier/keyword)
      if (/[a-zA-Z_]/.test(line[i])) {
        let word = '';
        while (i < line.length && /[a-zA-Z0-9_]/.test(line[i])) {
          word += line[i++];
        }
        tokens.push({ type: 'word', value: word });
        continue;
      }
      
      // Operator or punctuation
      tokens.push({ type: 'punct', value: line[i++] });
    }
    
    // Now render tokens with highlighting
    const keywords = new Set(['from', 'import', 'for', 'in', 'if', 'else', 'elif', 'def', 'class', 
                      'return', 'yield', 'with', 'as', 'try', 'except', 'finally', 
                      'raise', 'pass', 'break', 'continue', 'and', 'or', 'not', 'is',
                      'lambda', 'None', 'True', 'False']);
    const builtins = new Set(['print', 'range', 'len', 'str', 'int', 'float', 'list', 'dict', 
                      'set', 'tuple', 'zip', 'map', 'filter', 'enumerate', 'sorted',
                      'sum', 'min', 'max', 'abs', 'round', 'open', 'type', 'isinstance']);
    
    let result = '';
    for (let j = 0; j < tokens.length; j++) {
      const tok = tokens[j];
      const nextTok = tokens[j + 1];
      const nextNonWs = tokens.slice(j + 1).find(t => t.type !== 'ws');
      
      switch (tok.type) {
        case 'ws':
          result += tok.value;
          break;
        case 'string':
          result += '<span class="syn-string">' + this.escapeHtml(tok.value) + '</span>';
          break;
        case 'number':
          result += '<span class="syn-number">' + tok.value + '</span>';
          break;
        case 'word':
          if (keywords.has(tok.value)) {
            result += '<span class="syn-keyword">' + tok.value + '</span>';
          } else if (builtins.has(tok.value) && nextNonWs && nextNonWs.value === '(') {
            result += '<span class="syn-builtin">' + tok.value + '</span>';
          } else if (/^[A-Z]/.test(tok.value)) {
            result += '<span class="syn-class">' + tok.value + '</span>';
          } else if (nextNonWs && nextNonWs.value === '(') {
            result += '<span class="syn-function">' + tok.value + '</span>';
          } else {
            result += this.escapeHtml(tok.value);
          }
          break;
        case 'punct':
          if ('=+-*/<>'.includes(tok.value)) {
            result += '<span class="syn-operator">' + this.escapeHtml(tok.value) + '</span>';
          } else {
            result += this.escapeHtml(tok.value);
          }
          break;
        default:
          result += this.escapeHtml(tok.value);
      }
    }
    
    return result;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  loadPreset(preset) {
    const presets = {
      '2d': [
        { name: 'y', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
        { name: 'x', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
      ],
      '3d': [
        { name: 'z', type: 'space', unit: 'micrometer', scale: 2, translation: 0, scaleFactor: 2 },
        { name: 'y', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
        { name: 'x', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
      ],
      '4d': [
        { name: 'c', type: 'channel', unit: '', scale: 1, translation: 0, scaleFactor: 1 },
        { name: 'z', type: 'space', unit: 'micrometer', scale: 2, translation: 0, scaleFactor: 2 },
        { name: 'y', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
        { name: 'x', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
      ],
      '5d': [
        { name: 't', type: 'time', unit: 'second', scale: 1, translation: 0, scaleFactor: 1 },
        { name: 'c', type: 'channel', unit: '', scale: 1, translation: 0, scaleFactor: 1 },
        { name: 'z', type: 'space', unit: 'micrometer', scale: 2, translation: 0, scaleFactor: 2 },
        { name: 'y', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
        { name: 'x', type: 'space', unit: 'micrometer', scale: 0.5, translation: 0, scaleFactor: 2 },
      ],
    };
    this.dimensions = [...presets[preset]];
    this.validateDimensions();
  }

  addDimension() {
    this.dimensions = [
      ...this.dimensions,
      { name: 'dim', type: 'space', unit: '', scale: 1, translation: 0, scaleFactor: 1 },
    ];
    this.validateDimensions();
  }

  removeDimension(index) {
    this.dimensions = this.dimensions.filter((_, i) => i !== index);
    this.validateDimensions();
  }

  updateDimension(index, field, value) {
    const newDims = [...this.dimensions];
    newDims[index] = { ...newDims[index], [field]: value };
    this.dimensions = newDims;
    this.validateDimensions();
  }

  // Drag and drop methods for reordering dimensions
  handleDragStart(e, index) {
    this.draggedIndex = index;
    e.currentTarget.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', e.currentTarget.innerHTML);
  }

  handleDragEnd(e) {
    e.currentTarget.classList.remove('dragging');
    // Remove all drag-over classes
    this.shadowRoot.querySelectorAll('.drag-placeholder-top, .drag-placeholder-bottom').forEach(el => {
      el.classList.remove('drag-placeholder-top', 'drag-placeholder-bottom');
    });
    this.dropPosition = null;
  }

  handleDragOver(e, index) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';

    if (this.draggedIndex === index) {
      return;
    }

    // Calculate mouse position relative to the row
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const rowHeight = rect.height;
    const isTopHalf = mouseY < rowHeight / 2;

    // Remove previous drag-over indicators
    this.shadowRoot.querySelectorAll('.drag-placeholder-top, .drag-placeholder-bottom').forEach(el => {
      el.classList.remove('drag-placeholder-top', 'drag-placeholder-bottom');
    });

    // Add visual indicator for drop position
    if (isTopHalf) {
      e.currentTarget.classList.add('drag-placeholder-top');
      this.dropPosition = 'before';
    } else {
      e.currentTarget.classList.add('drag-placeholder-bottom');
      this.dropPosition = 'after';
    }
  }

  handleDrop(e, dropIndex) {
    e.preventDefault();
    e.stopPropagation();

    if (this.draggedIndex === null) {
      return;
    }

    // Calculate actual insert position based on drop position
    let insertIndex = dropIndex;

    if (this.dropPosition === 'after') {
      insertIndex = dropIndex + 1;
    }

    // Adjust if dragging from earlier position
    if (this.draggedIndex < insertIndex) {
      insertIndex--;
    }

    // Don't do anything if dropping in same position
    if (this.draggedIndex === insertIndex) {
      this.draggedIndex = null;
      this.dropPosition = null;
      return;
    }

    // Reorder dimensions
    const newDims = [...this.dimensions];
    const [draggedItem] = newDims.splice(this.draggedIndex, 1);
    newDims.splice(insertIndex, 0, draggedItem);

    this.dimensions = newDims;
    this.draggedIndex = null;
    this.dropPosition = null;
    this.validateDimensions();
  }

  generateJSON() {
    const axes = this.dimensions.map(d => {
      const axis = { name: d.name, type: d.type || undefined };
      if (d.unit) axis.unit = d.unit;
      return axis;
    });

    const datasets = [];
    for (let level = 0; level < this.numLevels; level++) {
      const scale = this.dimensions.map(d =>
        d.scale * Math.pow(d.scaleFactor || 1, level)
      );
      const transforms = [{ scale }];

      if (this.dimensions.some(d => d.translation !== 0)) {
        transforms.push({
          translation: this.dimensions.map(d => d.translation || 0)
        });
      }

      datasets.push({
        path: String(level),
        coordinateTransformations: transforms
      });
    }

    const multiscale = {
      version: this.version === 'v0.4' ? '0.4' : '0.5',
      name: 'example_image',
      axes,
      datasets,
    };

    if (this.version === 'v0.5') {
      // v0.5 uses coordinateTransformations at multiscale level too
      multiscale.coordinateTransformations = [{
        type: 'scale',
        scale: [1, 1, 1, 1, 1].slice(0, this.dimensions.length)
      }];
    }

    // v0.5: zarr.json with zarr_format, node_type, attributes
    // v0.4: .zattrs with just the multiscales array
    const output = this.version === 'v0.5'
      ? { 
          zarr_format: 3,
          node_type: 'group',
          attributes: {
            ome: { multiscales: [multiscale] }
          }
        }
      : { multiscales: [multiscale] };

    return this.compactJSON(output);
  }

  // Custom JSON formatter that keeps arrays on single lines
  compactJSON(obj, indent = 0) {
    const spaces = '  '.repeat(indent);
    const nextSpaces = '  '.repeat(indent + 1);
    
    if (obj === null) return 'null';
    if (typeof obj === 'boolean') return obj ? 'true' : 'false';
    if (typeof obj === 'number') return String(obj);
    if (typeof obj === 'string') return JSON.stringify(obj);
    
    if (Array.isArray(obj)) {
      // Check if this is a simple array (numbers, strings, or simple objects)
      const isSimple = obj.every(item => 
        typeof item === 'number' || 
        typeof item === 'string' ||
        typeof item === 'boolean' ||
        item === null
      );
      
      if (isSimple) {
        // Keep simple arrays on one line
        return '[' + obj.map(item => 
          typeof item === 'string' ? JSON.stringify(item) : String(item)
        ).join(', ') + ']';
      }
      
      // Complex arrays get one item per line
      if (obj.length === 0) return '[]';
      const items = obj.map(item => nextSpaces + this.compactJSON(item, indent + 1));
      return '[\n' + items.join(',\n') + '\n' + spaces + ']';
    }
    
    if (typeof obj === 'object') {
      const keys = Object.keys(obj);
      if (keys.length === 0) return '{}';
      
      const items = keys.map(key => {
        const value = this.compactJSON(obj[key], indent + 1);
        return nextSpaces + JSON.stringify(key) + ': ' + value;
      });
      return '{\n' + items.join(',\n') + '\n' + spaces + '}';
    }
    
    return String(obj);
  }

  generatePython() {
    const ver = this.version === 'v0.4' ? 'v04' : 'v05';

    const axesCode = this.dimensions.map(d => {
      const typeMap = {
        'space': 'SpaceAxis',
        'time': 'TimeAxis',
        'channel': 'ChannelAxis',
      };
      const axisClass = typeMap[d.type] || 'SpaceAxis';
      const unit = d.unit ? `, unit="${d.unit}"` : '';
      return `    ${ver}.${axisClass}(name="${d.name}"${unit})`;
    }).join(',\n');

    const dimsCode = this.dimensions.map(d => {
      const parts = [`name="${d.name}"`];
      if (d.type) parts.push(`type="${d.type}"`);
      if (d.unit) parts.push(`unit="${d.unit}"`);
      if (d.scale !== 1) parts.push(`scale=${d.scale}`);
      if (d.translation !== 0) parts.push(`translation=${d.translation}`);
      if (d.scaleFactor !== 1) parts.push(`scale_factor=${d.scaleFactor}`);
      return `    DimSpec(${parts.join(', ')})`;
    }).join(',\n');

    return `from yaozarrs import ${ver}, DimSpec

# Method 1: Using axis classes directly
axes = [
${axesCode}
]

# Create coordinate transformations manually
datasets = []
for level in range(${this.numLevels}):
    scale = [${this.dimensions.map(d => d.scale).join(', ')}]
    # Apply scale factors per level
    scale = [s * (sf ** level) for s, sf in zip(
        scale,
        [${this.dimensions.map(d => d.scaleFactor || 1).join(', ')}]
    )]
    transforms = [${ver}.Scale(scale=scale)]
    datasets.append(
        ${ver}.Dataset(
            path=str(level),
            coordinateTransformations=transforms
        )
    )

multiscale = ${ver}.Multiscale(
    name="example_image",
    axes=axes,
    datasets=datasets
)

# Method 2: Using DimSpec (simpler!)
dims = [
${dimsCode}
]

multiscale = ${ver}.Multiscale.from_dims(
    dims,
    name="example_image",
    n_levels=${this.numLevels}
)

# Convert to JSON
metadata = ${ver}.Metadata(multiscales=[multiscale])
print(metadata.model_dump_json(indent=2))`;
  }

  async copyToClipboard() {
    let text;
    if (this.activeTab === 'json') {
      text = this.generateJSONForSelectedNode();
      if (text === null) {
        return; // Nothing to copy
      }
    } else {
      text = this.generatePython();
    }
    await navigator.clipboard.writeText(text);

    // Visual feedback
    this.copyButtonText = 'Copied!';
    this.requestUpdate();
    setTimeout(() => {
      this.copyButtonText = 'Copy';
      this.requestUpdate();
    }, 2000);
  }

  getHighlightedCode() {
    if (this.activeTab === 'json') {
      const json = this.generateJSONForSelectedNode();
      if (json === null) {
        return '<span class="syn-comment">// Select a metadata file (.zattrs, .zarray, .zgroup, or zarr.json)\n// to see its contents</span>';
      }
      return this.highlightJSON(json);
    } else {
      return this.highlightPython(this.generatePython());
    }
  }

  // Returns true if the selected node is a metadata file
  isMetadataFile(nodeId) {
    if (!nodeId) return false;
    // root-meta is .zattrs (v2) or zarr.json (v3) for the root group
    // root-zgroup is .zgroup (v2 only)
    // level-N-meta is .zarray (v2) or zarr.json (v3) for arrays
    return nodeId === 'root-meta' || 
           nodeId === 'root-zgroup' || 
           nodeId.endsWith('-meta');
  }

  generateJSONForSelectedNode() {
    const nodeId = this.selectedNode;
    if (!this.isMetadataFile(nodeId)) {
      return null;
    }

    if (nodeId === 'root-meta') {
      // Root group metadata (.zattrs or zarr.json)
      return this.generateJSON();
    }

    if (nodeId === 'root-zgroup') {
      // .zgroup file (v2 only)
      return this.compactJSON({ zarr_format: 2 });
    }

    // Array metadata (level-N-meta)
    const match = nodeId.match(/^level-(\d+)-meta$/);
    if (match) {
      const level = parseInt(match[1]);
      return this.generateArrayJSON(level);
    }

    return null;
  }

  generateArrayJSON(level) {
    const isV05 = this.version === 'v0.5';
    
    // Calculate shape for this level (example base shape, scaled down)
    const baseShapes = { c: 3, t: 10, z: 100, y: 1024, x: 1024 };
    const shape = this.dimensions.map(d => {
      const base = baseShapes[d.name] || 256;
      if (d.type === 'space') {
        return Math.max(1, Math.floor(base / Math.pow(d.scaleFactor || 2, level)));
      }
      return base;
    });

    const chunks = this.dimensions.map(d => {
      if (d.type === 'space') return 64;
      if (d.type === 'channel') return 1;
      if (d.type === 'time') return 1;
      return 64;
    });

    if (isV05) {
      // Zarr v3 array metadata
      return this.compactJSON({
        zarr_format: 3,
        node_type: 'array',
        shape: shape,
        data_type: 'uint16',
        chunk_grid: {
          name: 'regular',
          configuration: { chunk_shape: chunks }
        },
        chunk_key_encoding: {
          name: 'default',
          configuration: { separator: '/' }
        },
        fill_value: 0,
        codecs: '...',
        dimension_names: this.dimensions.map(d => d.name)
      });
    } else {
      // Zarr v2 .zarray
      return this.compactJSON({
        zarr_format: 2,
        shape: shape,
        chunks: chunks,
        dtype: '<u2',
        compressor: '...',
        fill_value: 0,
        order: 'C',
        filters: null,
        dimension_separator: '/'
      });
    }
  }

  // Tree view methods
  toggleNode(nodeId) {
    this.expandedNodes = {
      ...this.expandedNodes,
      [nodeId]: !this.expandedNodes[nodeId]
    };
  }

  selectNode(nodeId) {
    this.selectedNode = nodeId;
  }

  getNodeInfo(nodeId) {
    const isV05 = this.version === 'v0.5';
    const metaFile = isV05 ? 'zarr.json' : '.zattrs';
    const arrayMeta = isV05 ? 'zarr.json' : '.zarray';
    
    const infos = {
      'root': {
        title: 'image.zarr/',
        desc: `Root group of the OME-Zarr image. Contains ${isV05 ? 'zarr.json with OME metadata under attributes.ome' : '.zattrs with multiscales metadata'}.`
      },
      'root-meta': {
        title: metaFile,
        desc: isV05 
          ? 'Zarr v3 group metadata file. Contains zarr_format, node_type, and OME metadata under attributes.ome.multiscales.'
          : 'Zarr v2 attributes file containing the multiscales array with axes and coordinate transformations.'
      },
      'root-zgroup': {
        title: '.zgroup',
        desc: 'Zarr v2 group marker file. Contains {"zarr_format": 2} to identify this as a Zarr group.'
      },
    };

    // Add level info dynamically
    for (let i = 0; i < this.numLevels; i++) {
      const isFirst = i === 0;
      const levelDesc = isFirst ? 'Full resolution' : `${Math.pow(2, i)}× downsampled`;
      infos[`level-${i}`] = {
        title: `${i}/`,
        desc: `${levelDesc} pyramid level. Contains the array data as chunked storage.`
      };
      infos[`level-${i}-meta`] = {
        title: arrayMeta,
        desc: isV05
          ? `Zarr v3 array metadata. Defines shape, chunks, dtype, codecs, and dimension_names matching the axes: [${this.dimensions.map(d => `"${d.name}"`).join(', ')}].`
          : `Zarr v2 array metadata. Defines shape, chunks, dtype, compressor, and dimension_separator.`
      };
      infos[`level-${i}-chunks`] = {
        title: this.dimensions.map(d => d.name).join('/') + '/...',
        desc: `Chunk files organized by dimension. Each chunk contains a portion of the array data compressed according to the codec settings.`
      };
    }

    return infos[nodeId] || { title: nodeId, desc: 'Select an item to see its description.' };
  }

  renderTreeNode(nodeId, label, icon, isExpandable, children = null, depth = 0) {
    const isExpanded = this.expandedNodes[nodeId];
    const isSelected = this.selectedNode === nodeId;
    
    return html`
      <div class="tree-node">
        <div 
          class="tree-item ${isSelected ? 'selected' : ''}"
          @click=${() => this.selectNode(nodeId)}
        >
          <span 
            class="tree-toggle ${isExpandable ? 'expandable' : ''}"
            @click=${(e) => { if (isExpandable) { e.stopPropagation(); this.toggleNode(nodeId); }}}
          >
            ${isExpandable ? (isExpanded ? '▼' : '▶') : ''}
          </span>
          <span class="tree-icon ${icon}">${icon === 'folder' ? '📁' : icon === 'file' ? '📄' : '📦'}</span>
          <span class="tree-label">${label}</span>
        </div>
        ${children && isExpandable ? html`
          <div class="tree-children ${isExpanded ? '' : 'collapsed'}">
            ${children}
          </div>
        ` : ''}
      </div>
    `;
  }

  renderTree() {
    const isV05 = this.version === 'v0.5';
    const metaFile = isV05 ? 'zarr.json' : '.zattrs';
    const arrayMeta = isV05 ? 'zarr.json' : '.zarray';
    const chunkPath = this.dimensions.map(d => d.name).join('/');

    const levelNodes = [];
    for (let i = 0; i < this.numLevels; i++) {
      const isLast = i === this.numLevels - 1;
      levelNodes.push(
        this.renderTreeNode(`level-${i}`, `${i}/`, 'folder', true, html`
          ${this.renderTreeNode(`level-${i}-meta`, arrayMeta, 'file', false)}
          ${this.renderTreeNode(`level-${i}-chunks`, chunkPath + '/...', 'chunk', false)}
        `)
      );
    }

    return html`
      <div class="tree-info-panel">
        ${this.selectedNode ? html`
          <div class="tree-info-title">${this.getNodeInfo(this.selectedNode).title}</div>
          <div class="tree-info-desc">${this.getNodeInfo(this.selectedNode).desc}</div>
        ` : html`
          <div class="tree-info-hint">Click a file or folder to see its purpose</div>
        `}
      </div>
      <div class="tree-view">
        ${this.renderTreeNode('root', 'image.zarr/', 'folder', true, html`
          ${this.renderTreeNode('root-meta', metaFile, 'file', false)}
          ${!isV05 ? this.renderTreeNode('root-zgroup', '.zgroup', 'file', false) : ''}
          ${levelNodes}
        `)}
      </div>
    `;
  }

  render() {
    return html`
      <div class="explorer-container">
        <div class="toolbar">
          <div class="toolbar-group">
            <label>Presets</label>
            <button class="preset-btn" @click=${() => this.loadPreset('2d')}>2D</button>
            <button class="preset-btn" @click=${() => this.loadPreset('3d')}>3D</button>
            <button class="preset-btn" @click=${() => this.loadPreset('4d')}>4D</button>
            <button class="preset-btn" @click=${() => this.loadPreset('5d')}>5D</button>
          </div>
          <div class="toolbar-separator"></div>
          <div class="toolbar-group">
            <label>Pyramid Levels</label>
            <input
              class="levels-input"
              type="number"
              min="1"
              max="10"
              .value=${this.numLevels}
              @input=${(e) => this.numLevels = parseInt(e.target.value)}
            />
          </div>
        </div>

        <div class="main-content">
          <div class="input-panel">
            <div class="settings-group">
              <table class="dimension-table">
                <thead>
                  <tr>
                    <th></th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Unit</th>
                    <th>Scale</th>
                    <th>Translate</th>
                    <th>Downscale Factor</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  ${this.dimensions.map((dim, i) => html`
                    <tr
                      draggable="true"
                      @dragstart=${(e) => this.handleDragStart(e, i)}
                      @dragend=${(e) => this.handleDragEnd(e)}
                      @dragover=${(e) => this.handleDragOver(e, i)}
                      @drop=${(e) => this.handleDrop(e, i)}
                    >
                      <td class="drag-handle" title="Drag to reorder">⋮⋮</td>
                      <td>
                        <input
                          type="text"
                          .value=${dim.name}
                          @input=${(e) => this.updateDimension(i, 'name', e.target.value)}
                        />
                      </td>
                      <td>
                        <select
                          .value=${dim.type}
                          @change=${(e) => this.updateDimension(i, 'type', e.target.value)}
                        >
                          <option value="space">space</option>
                          <option value="time">time</option>
                          <option value="channel">channel</option>
                        </select>
                      </td>
                      <td>
                        <input
                          type="text"
                          list="units-${i}"
                          .value=${dim.unit}
                          @input=${(e) => this.updateDimension(i, 'unit', e.target.value)}
                          placeholder="—"
                          title="${this.getValidUnits(dim.type).length > 0 ? 'Start typing for suggestions' : 'Any unit allowed'}"
                        />
                        <datalist id="units-${i}">
                          ${this.getValidUnits(dim.type).map(unit => html`
                            <option value="${unit}"></option>
                          `)}
                        </datalist>
                      </td>
                      <td>
                        <input
                          type="number"
                          step="any"
                          .value=${dim.scale}
                          @input=${(e) => this.updateDimension(i, 'scale', parseFloat(e.target.value) || 0)}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          step="any"
                          .value=${dim.translation}
                          @input=${(e) => this.updateDimension(i, 'translation', parseFloat(e.target.value) || 0)}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          step="any"
                          .value=${dim.scaleFactor}
                          @input=${(e) => this.updateDimension(i, 'scaleFactor', parseFloat(e.target.value) || 1)}
                        />
                      </td>
                      <td>
                        <button class="delete-btn" @click=${() => this.removeDimension(i)} title="Remove dimension">✕</button>
                      </td>
                    </tr>
                  `)}
                </tbody>
              </table>
              <button class="add-dimension primary" @click=${() => this.addDimension()}>
                <span>+</span> Add Dimension
              </button>
            </div>
            ${this.validationErrors.length > 0 ? html`
              <div class="validation-panel ${this.validationErrors.some(e => e.type === 'error') ? 'error' : ''}">
                <div class="validation-title">
                  <span class="validation-icon">${this.validationErrors.some(e => e.type === 'error') ? '⚠️' : '💡'}</span>
                  ${this.validationErrors.some(e => e.type === 'error') ? 'Spec Violations Detected' : 'Warnings'}
                </div>
                <ul class="validation-list">
                  ${this.validationErrors.map(err => html`
                    <li>
                      <strong>${err.message}</strong>
                      ${err.hint ? html`<div class="validation-hint">${err.hint}</div>` : ''}
                    </li>
                  `)}
                </ul>
              </div>
            ` : ''}
          </div>

          <div class="output-panel">
            <div class="tab-bar">
              <button
                class="tab ${this.activeTab === 'json' ? 'active' : ''}"
                @click=${() => this.activeTab = 'json'}
              >
                Spec JSON
              </button>
              <button
                class="tab ${this.activeTab === 'python' ? 'active' : ''}"
                @click=${() => this.activeTab = 'python'}
              >
                Python
              </button>
              <div class="tab-spacer"></div>
              <div class="version-toggle">
                <button
                  class=${this.version === 'v0.5' ? 'active' : ''}
                  @click=${() => this.version = 'v0.5'}
                >
                  v0.5
                </button>
                <button
                  class=${this.version === 'v0.4' ? 'active' : ''}
                  @click=${() => this.version = 'v0.4'}
                >
                  v0.4
                </button>
              </div>
            </div>

            <div class="code-area">
              <div class="tree-panel">
                ${this.renderTree()}
              </div>

              <div class="code-output">
                <div class="tab-content">
                  <button 
                    class="copy-button ${this.copyButtonText.includes('Copied') ? 'copied' : ''}" 
                    @click=${this.copyToClipboard}
                  >
                    ${this.copyButtonText.includes('Copied') ? '✓' : '📋'} ${this.copyButtonText}
                  </button>
                  <pre class="code-block">${unsafeHTML(this.getHighlightedCode())}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }
}

customElements.define('ome-explorer', OmeExplorer);
