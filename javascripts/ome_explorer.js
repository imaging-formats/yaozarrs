/**
 * Interactive OME-Zarr Metadata Explorer
 * Educational tool for understanding OME-NGFF metadata structure
 */

import { LitElement, html, css } from 'https://cdn.jsdelivr.net/npm/lit@3.1.0/+esm';
import { unsafeHTML } from 'https://cdn.jsdelivr.net/npm/lit@3.1.0/directives/unsafe-html.js/+esm';
import {
  generateJSON,
  generatePython,
  generateArrayMetadataJSON,
  generatePlateJSON,
  generateWellJSON,
  generatePlatePython,
  compactJSON,
  getValidUnits,
  validateDimensions,
} from './ome_generator.js';

/**
 * Dimension presets for different image types
 */
const PRESETS = {
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

/**
 * ZarrTreeViewer - Reusable component for displaying Zarr file structure
 * with collapsible JSON viewer
 */
class ZarrTreeViewer extends LitElement {
  static properties = {
    treeData: { type: Object },
    fileContents: { type: Object },
    activeTab: { type: String },
    copyButtonText: { type: String },
    expandedNodes: { type: Object },
    selectedNode: { type: String },
    collapsedJsonPaths: { type: Object },
  };

  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      font-family: var(--md-text-font-family, 'Inter', -apple-system, BlinkMacSystemFont, sans-serif);
      --primary-color: #4051b5;
      --code-bg: #1e1e2e;
      --code-bg-lighter: #262637;

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

    .viewer-container {
      display: flex;
      flex-direction: row;
      min-height: 400px;
      background: var(--code-bg);
      border-radius: 4px;
      overflow: hidden;
    }

    .tree-panel {
      display: flex;
      flex-direction: column;
      background: var(--code-bg);
      border-right: 1px solid rgba(255, 255, 255, 0.1);
      width: 220px;
      flex-shrink: 0;
      overflow-y: auto;
      padding: 0.5rem;
    }

    .tree-node {
      user-select: none;
    }

    .tree-item {
      display: flex;
      align-items: center;
      padding: 0.125rem 0.25rem;
      border-radius: 3px;
      color: #cdd6f4;
      font-size: 0.75rem;
      cursor: pointer;
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
      margin-right: 0.125rem;
      color: #9399b2;
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
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
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
      font-size: 0.6875rem;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      margin-top: auto;
      line-height: 1.4;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      max-height: 180px;
      overflow-y: auto;
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
      padding-top: 0.25rem;
    }

    .code-output {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }

    .code-block {
      font-family: var(--md-code-font-family, 'Fira Code', -apple-system, BlinkMacSystemFont, sans-serif);
      font-size: 0.6875rem;
      line-height: 1.6;
      color: #cdd6f4;
      background: var(--code-bg-lighter);
      padding: 1rem;
      margin: 0;
      overflow: auto;
      flex: 1;
      width: 100%;
      box-sizing: border-box;
    }

    .copy-button {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      padding: 0.25rem 0.5rem;
      background: rgba(0, 0, 0, 0.3);
      color: #cdd6f4;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.6875rem;
      display: flex;
      align-items: center;
      gap: 0.25rem;
      transition: all 0.15s;
      z-index: 10;
    }

    .copy-button:hover {
      background: rgba(0, 0, 0, 0.5);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .copy-button.copied {
      background: rgba(16, 185, 129, 0.2);
      border-color: #10b981;
      color: #10b981;
    }

    .tab-content {
      position: relative;
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
    }

    /* Syntax highlighting */
    .syn-keyword { color: var(--syn-keyword); }
    .syn-string { color: var(--syn-string); }
    .syn-number { color: var(--syn-number); }
    .syn-comment { color: var(--syn-comment); }
    .syn-function { color: var(--syn-function); }
    .syn-property { color: var(--syn-property); }
    .syn-punctuation { color: var(--syn-punctuation); }
    .syn-operator { color: var(--syn-operator); }
    .syn-builtin { color: var(--syn-builtin); }
    .syn-constant { color: var(--syn-constant); }

    /* Collapsible JSON */
    .json-line {
      display: flex;
      align-items: flex-start;
      font-family: var(--md-code-font-family, 'Fira Code', -apple-system, BlinkMacSystemFont, sans-serif);
    }

    .json-fold {
      cursor: pointer;
      user-select: none;
      color: #9399b2;
      margin-right: 0.125rem;
      width: 12px;
      flex-shrink: 0;
      text-align: center;
      font-size: 0.625rem;
      line-height: 1.5;
      transition: color 0.15s;
    }

    .json-fold:hover {
      color: var(--primary-color);
    }

    .json-fold.collapsed::before {
      content: '▶';
    }

    .json-fold.expanded::before {
      content: '▼';
    }

    .json-fold.empty {
      cursor: default;
      opacity: 0;
    }

    .json-content {
      flex: 1;
      min-width: 0;
    }

    .json-children {
      margin-left: 0.625rem;
    }

    .json-children.collapsed {
      display: none;
    }

    .json-ellipsis {
      color: #6c7086;
      font-style: italic;
    }
  `;

  constructor() {
    super();
    this.treeData = null;
    this.fileContents = {};
    this.activeTab = 'json';
    this.copyButtonText = 'Copy';
    this.expandedNodes = {};
    this.selectedNode = null;
    this.collapsedJsonPaths = {};
  }

  toggleNode(nodeId) {
    this.expandedNodes = {
      ...this.expandedNodes,
      [nodeId]: !this.expandedNodes[nodeId]
    };
  }

  selectNode(nodeId) {
    this.selectedNode = nodeId;
    this.dispatchEvent(new CustomEvent('node-selected', {
      detail: { nodeId },
      bubbles: true,
      composed: true
    }));
  }

  toggleJsonPath(path) {
    this.collapsedJsonPaths = {
      ...this.collapsedJsonPaths,
      [path]: !this.collapsedJsonPaths[path]
    };
  }

  renderCollapsibleJSON(obj, path = '', indent = 0, trailingComma = '') {
    if (obj === null) return html`<span class="syn-constant">null${trailingComma}</span>`;
    if (typeof obj === 'boolean') return html`<span class="syn-constant">${obj}${trailingComma}</span>`;
    if (typeof obj === 'number') return html`<span class="syn-number">${obj}${trailingComma}</span>`;
    if (typeof obj === 'string') return html`<span class="syn-string">"${obj}"${trailingComma}</span>`;

    const isCollapsed = this.collapsedJsonPaths[path];

    if (Array.isArray(obj)) {
      const isEmpty = obj.length === 0;
      if (isEmpty) {
        return html`[]${trailingComma}`;
      }

      // Check if it's a simple array (primitives only)
      const isSimple = obj.every(item =>
        typeof item === 'number' ||
        typeof item === 'string' ||
        typeof item === 'boolean' ||
        item === null
      );

      if (isSimple) {
        const content = obj.map((item, i) => {
          const comma = i < obj.length - 1 ? ', ' : '';
          if (typeof item === 'string') return `"${item}"${comma}`;
          if (typeof item === 'number') return html`<span class="syn-number">${item}${comma}</span>`;
          if (typeof item === 'boolean' || item === null) return html`<span class="syn-constant">${item}${comma}</span>`;
          return item;
        });
        return html`[${content}]${trailingComma}`;
      }

      return html`
        <div>
          <div class="json-line">
            <span
              class="json-fold ${isCollapsed ? 'collapsed' : 'expanded'}"
              @click=${() => this.toggleJsonPath(path)}
            ></span>
            <span class="json-content">
              [${isCollapsed ? html`<span class="json-ellipsis"> /* ${obj.length} items */ </span>]${trailingComma}` : ''}
            </span>
          </div>
          ${!isCollapsed ? html`
            <div class="json-children">
              ${obj.map((item, i) => {
                const comma = i < obj.length - 1 ? ',' : '';
                const itemContent = this.renderCollapsibleJSON(item, `${path}[${i}]`, indent + 1, comma);

                // Check if item is a primitive (single-line)
                const isPrimitive = typeof item !== 'object' || item === null ||
                  (Array.isArray(item) && item.every(x => typeof x !== 'object' || x === null)) ||
                  (typeof item === 'object' && Object.keys(item).length === 0);

                // Check if item is a simple object (all values are primitives)
                const isSimpleObject = typeof item === 'object' && item !== null && !Array.isArray(item) &&
                  Object.keys(item).length > 0 &&
                  Object.keys(item).every(key => {
                    const value = item[key];
                    return (
                      typeof value === 'number' ||
                      typeof value === 'string' ||
                      typeof value === 'boolean' ||
                      value === null
                    );
                  });

                // Render primitives and simple objects on their own line
                if (isPrimitive || isSimpleObject) {
                  return html`
                    <div class="json-line">
                      <span class="json-fold empty"></span>
                      <span class="json-content">${itemContent}</span>
                    </div>
                  `;
                } else {
                  // For complex items, just render them directly and append comma to closing bracket
                  return html`${itemContent}`;
                }
              })}
            </div>
            <div class="json-line">
              <span class="json-fold empty"></span>
              <span class="json-content">]${trailingComma}</span>
            </div>
          ` : ''}
        </div>
      `;
    }

    if (typeof obj === 'object') {
      const keys = Object.keys(obj);
      const isEmpty = keys.length === 0;

      if (isEmpty) {
        return html`{}${trailingComma}`;
      }

      return html`
        <div>
          <div class="json-line">
            <span
              class="json-fold ${isCollapsed ? 'collapsed' : 'expanded'}"
              @click=${() => this.toggleJsonPath(path)}
            ></span>
            <span class="json-content">
              {${isCollapsed ? html`<span class="json-ellipsis"> /* ${keys.length} ${keys.length === 1 ? 'property' : 'properties'} */ </span>}${trailingComma}` : ''}
            </span>
          </div>
          ${!isCollapsed ? html`
            <div class="json-children">
              ${keys.map((key, i) => {
                const value = obj[key];
                const valuePath = path ? `${path}.${key}` : key;
                const comma = i < keys.length - 1 ? ',' : '';
                const valueContent = this.renderCollapsibleJSON(value, valuePath, indent + 1, comma);

                // Check if value is a primitive (single-line)
                const isPrimitive = typeof value !== 'object' || value === null ||
                  (Array.isArray(value) && value.every(x => typeof x !== 'object' || x === null)) ||
                  (typeof value === 'object' && Object.keys(value).length === 0);

                return html`
                  <div class="json-line">
                    <span class="json-fold empty"></span>
                    <span class="json-content">
                      <span class="syn-property">"${key}"</span>: ${isPrimitive ? valueContent : html`<div style="display: inline-block; vertical-align: top; width: 100%;">${valueContent}</div>`}
                    </span>
                  </div>
                `;
              })}
            </div>
            <div class="json-line">
              <span class="json-fold empty"></span>
              <span class="json-content">}${trailingComma}</span>
            </div>
          ` : ''}
        </div>
      `;
    }

    return html`${String(obj)}${trailingComma}`;
  }

  renderTreeNode(node) {
    if (!node) return '';

    const isExpanded = this.expandedNodes[node.id];
    const isSelected = this.selectedNode === node.id;
    const hasChildren = node.children && node.children.length > 0;

    return html`
      <div class="tree-node">
        <div
          class="tree-item ${isSelected ? 'selected' : ''}"
          @click=${() => this.selectNode(node.id)}
        >
          <span
            class="tree-toggle ${hasChildren ? 'expandable' : ''}"
            @click=${(e) => { if (hasChildren) { e.stopPropagation(); this.toggleNode(node.id); }}}
          >
            ${hasChildren ? (isExpanded ? '▼' : '▶') : ''}
          </span>
          <span class="tree-icon ${node.icon}">${node.icon === 'folder' ? '📁' : node.icon === 'file' ? '📄' : '📦'}</span>
          <span class="tree-label">${node.label}</span>
        </div>
        ${hasChildren && node.children ? html`
          <div class="tree-children ${isExpanded ? '' : 'collapsed'}">
            ${node.children.map(child => this.renderTreeNode(child))}
          </div>
        ` : ''}
      </div>
    `;
  }

  copyToClipboard() {
    const selectedFile = this.fileContents[this.selectedNode];
    if (!selectedFile) return;

    const text = this.activeTab === 'json'
      ? compactJSON(selectedFile.content)
      : selectedFile.content;

    navigator.clipboard.writeText(text).then(() => {
      this.copyButtonText = 'Copied!';
      setTimeout(() => {
        this.copyButtonText = 'Copy';
      }, 2000);
    });
  }

  render() {
    const selectedFile = this.fileContents[this.selectedNode] || {};
    const content = selectedFile.content;

    return html`
      <div class="viewer-container">
        <div class="tree-panel">
          ${this.treeData ? this.renderTreeNode(this.treeData) : ''}
          ${selectedFile.title ? html`
            <div class="tree-info-panel">
              <div class="tree-info-title">${selectedFile.title}</div>
              <div class="tree-info-desc">${selectedFile.description || 'No description available.'}</div>
            </div>
          ` : ''}
        </div>

        <div class="code-output">
          <div class="tab-content">
            <button
              class="copy-button ${this.copyButtonText.includes('Copied') ? 'copied' : ''}"
              @click=${this.copyToClipboard}
            >
              ${this.copyButtonText.includes('Copied') ? '✓' : '📋'} ${this.copyButtonText}
            </button>
            <div class="code-block">
              ${content ? (
                typeof content === 'object'
                  ? this.renderCollapsibleJSON(content)
                  : html`<pre>${content}</pre>`
              ) : 'Select a file to view its contents'}
            </div>
          </div>
        </div>
      </div>
    `;
  }
}

customElements.define('zarr-tree-viewer', ZarrTreeViewer);

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
    collapsedJsonPaths: { type: Object },
    tooltips: { type: Object },
    preset: { type: String },
    levels: { type: Number },
    // Plate properties
    plateControl: {
      type: Boolean,
      attribute: 'plate-control',
      converter: {
        fromAttribute: (value) => {
          if (value === null) return undefined; // Use default
          if (value === 'false') return false;
          return true;
        }
      }
    },
    plateExpanded: {
      type: Boolean,
      attribute: 'plate-expanded',
      converter: {
        fromAttribute: (value) => {
          if (value === null) return undefined; // Use default
          if (value === 'false') return false;
          return true;
        }
      }
    },
    plateEnabled: { type: Boolean },
    plateType: { type: String },
    selectedWells: { type: Array },
    numFOVs: { type: Number },
  };

  updated(changedProperties) {
    if (changedProperties.has('version')) {
      // When version changes, validate that selected node still exists
      if (this.selectedNode === 'root-zgroup' && this.version === 'v0.5') {
        // .zgroup doesn't exist in v0.5, switch to root-meta
        this.selectedNode = 'root-meta';
      }
    }

    if (changedProperties.has('preset')) {
      // When preset changes, update dimensions
      this.loadPreset(this.preset);
    }

    if (changedProperties.has('levels')) {
      // When levels changes, update numLevels
      this.numLevels = this.levels;
    }

    // When plate configuration changes, refresh the selected node if needed
    if (changedProperties.has('plateType') || changedProperties.has('numFOVs')) {
      // If we're in plate mode with wells selected, ensure plate metadata is visible
      if (this.plateEnabled && this.selectedWells.length > 0) {
        // If the current selection is now invalid (e.g., a well that no longer exists),
        // fall back to root plate metadata
        if (this.selectedNode && this.selectedNode.startsWith('well-')) {
          // Validate that the well still exists
          const parts = this.selectedNode.split('-');
          if (parts.length >= 3) {
            const row = parseInt(parts[1]);
            const col = parseInt(parts[2]);
            const wellExists = this.selectedWells.some(w => w.row === row && w.col === col);
            if (!wellExists) {
              this.selectedNode = 'root-plate-meta';
            }
          }
        }
      }
    }
  }

  static styles = css`
    :host {
      display: block;
      font-family: var(--md-text-font-family, 'Inter', -apple-system, BlinkMacSystemFont, sans-serif);
      --primary-color: #4051b5;
      --primary-color-light: #5c6bc0;
      --accent-color: var(--md-accent-fg-color, #526cfe);
      --bg-color: var(--md-default-bg-color, #fff);
      --input-bg: var(--md-default-bg-color, #fff);
      --code-bg: #1e1e2e;
      --code-bg-lighter: #262637;
      --success-color: #10b981;
      --warning-color: #f59e0b;
    }

    /* Light mode specific colors */
    :host([data-theme="default"]),
    :host(:not([data-theme])) {
      --border-color: rgba(0, 0, 0, 0.08);
      --border-color-strong: rgba(0, 0, 0, 0.15);
      --text-muted: #64748b;
      --input-shadow: rgba(64, 81, 181, 0.15);
    }

    /* Dark mode specific colors */
    :host([data-theme="slate"]) {
      --border-color: rgba(255, 255, 255, 0.08);
      --border-color-strong: rgba(255, 255, 255, 0.15);
      --text-muted: #94a3b8;
      --input-bg: hsl(232, 15%, 18%);
      --input-shadow: rgba(137, 180, 250, 0.2);
    }

    :host {
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
      container-type: inline-size;
      border: 1px solid var(--border-color-strong);
      border-radius: 8px;
      overflow: hidden;
      background: var(--bg-color);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }

    .toolbar {
      display: flex;
      gap: 0.625rem;
      padding: 0.4rem 0.625rem;
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
      gap: 0.3rem;
      align-items: center;
    }

    .toolbar-group label {
      font-size: 0.5625rem;
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
      padding: 0.25rem 0.4rem;
      border: 1px solid var(--border-color-strong);
      border-radius: 4px;
      background: var(--input-bg);
      cursor: pointer;
      font-size: 0.6875rem;
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
      background: var(--primary-color);
      color: white;
      border-color: transparent;
      font-weight: 500;
      box-shadow: 0 1px 2px rgba(64, 81, 181, 0.2);
    }

    button.primary:hover {
      background: var(--primary-color-light);
    }

    .preset-btn {
      font-weight: 500;
    }

    .levels-input {
      width: 44px !important;
      text-align: center;
      font-weight: 500;
      padding: 0.25rem 0.2rem !important;
    }

    .main-content {
      display: flex;
      flex-direction: column;
    }

    .input-panel {
      border-bottom: 1px solid var(--border-color);
      padding: 0.5rem 0.625rem;
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
      font-size: 0.6875rem;
    }

    .dimension-table thead {
      position: relative;
    }

    .dimension-table th {
      text-align: left;
      padding: 0.3rem 0.4rem;
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.04);
      font-weight: 600;
      font-size: 0.5625rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-color-strong);
      white-space: nowrap;
      position: relative;
      overflow: visible;
    }

    .dimension-table th:first-child {
      border-radius: 4px 0 0 0;
    }

    .dimension-table th:last-child {
      border-radius: 0 4px 0 0;
    }

    .dimension-table td {
      padding: 0.2rem 0.3rem;
      border-top: 3px solid transparent;
      border-bottom: 3px solid transparent;
      vertical-align: middle;
    }

    .dimension-table tbody td {
      border-bottom-width: 1px;
      border-bottom-color: var(--border-color);
      border-bottom-style: solid;
      padding-bottom: calc(0.25rem + 2px);
    }

    .dimension-table tbody tr:last-child td {
      border-bottom: 3px solid transparent;
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
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.04);
    }

    .dimension-table tbody tr.drag-placeholder-top td {
      border-top-color: var(--primary-color);
    }

    .dimension-table tbody tr.drag-placeholder-bottom {
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.04);
    }

    .dimension-table tbody tr.drag-placeholder-bottom td {
      border-bottom-width: 3px;
      border-bottom-color: var(--primary-color);
      padding-bottom: 0.25rem;
    }

    @keyframes pulse-border-top {
      0%, 100% {
        border-top-width: 3px;
        box-shadow: none;
      }
      50% {
        border-top-width: 4px;
        box-shadow: 0 -2px 12px rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.4);
      }
    }

    @keyframes pulse-border-bottom {
      0%, 100% {
        border-bottom-width: 3px;
        box-shadow: none;
      }
      50% {
        border-bottom-width: 4px;
        box-shadow: 0 2px 12px rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.4);
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
      padding: 0.2rem 0.3rem;
      font-size: 0.6875rem;
      border-radius: 3px;
      border: 1px solid var(--border-color);
      background: var(--input-bg);
      transition: all 0.15s ease;
      box-sizing: border-box;
    }

    /* Name input - compact */
    .dimension-table td:nth-child(2) input {
      min-width: 50px;
    }

    /* Type select - fit "channel" */
    .dimension-table td:nth-child(3) select {
      min-width: 90px;
    }

    /* Unit input - fit "micrometer" */
    .dimension-table td:nth-child(4) input {
      min-width: 105px;
    }

    .dimension-table input:focus,
    .dimension-table select:focus {
      outline: none;
      border-color: var(--primary-color-light);
      box-shadow: 0 0 0 1px var(--input-shadow);
    }

    .dimension-table input[type="number"] {
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
      padding: 0.3rem 0.4rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.25rem;
      font-size: 0.6875rem;
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
      max-height: 600px;
      overflow: hidden;
    }

    .code-block {
      background: transparent;
      color: #cdd6f4;
      padding: 0.75rem;
      margin: 0;
      overflow: auto;
      flex: 1;
      width: 100%;
      box-sizing: border-box;
      font-family: var(--md-code-font-family, 'Fira Code', -apple-system, BlinkMacSystemFont, sans-serif);
      font-size: 0.6875rem;
      line-height: 1.5;
      min-height: 0;
      max-height: 600px;
    }

    .code-block pre {
      white-space: pre;
      margin: 0;
    }

    .copy-button {
      position: absolute;
      top: 0.5rem;
      right: 1rem;
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
      min-height: 400px;
      max-height: 600px;
    }

    .python-output {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: var(--code-bg);
      min-width: 0;
      overflow: hidden;
    }

    .tree-panel {
      display: flex;
      flex-direction: column;
      background: var(--code-bg);
      border-right: 1px solid rgba(255, 255, 255, 0.1);
      flex: 0 0 280px;
      min-width: 200px;
      max-height: 600px;
    }

    .tree-view {
      flex: 1;
      background: var(--code-bg);
      padding: 0.5rem;
      font-family: var(--md-code-font-family, 'Fira Code', -apple-system, BlinkMacSystemFont, sans-serif);
      font-size: 0.6875rem;
      overflow-y: auto;
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
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      height: 180px;
      max-height: 180px;
      overflow-y: auto;
      flex-shrink: 0;
    }

    .code-output {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      width: 100%;
      max-height: 600px;
      overflow: hidden;
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

    /* Tooltip Styles */
    .header-with-tooltip {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .info-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-left: 0.25rem;
      width: 14px;
      height: 14px;
      color: #89b4fa;
      cursor: help;
      position: relative;
      transition: all 0.2s ease;
      z-index: 100000;
    }

    .info-icon svg {
      width: 100%;
      height: 100%;
      transition: all 0.2s ease;
    }

    .info-icon:hover {
      color: #b4c4fa;
      transform: scale(1.15);
      z-index: 100001;
    }

    .tooltip {
      position: absolute;
      top: calc(100% + 8px);
      left: 50%;
      transform: translateX(-50%);
      background: rgba(30, 30, 46, 0.98);
      color: #cdd6f4;
      padding: 0.5rem 0.625rem;
      border-radius: 4px;
      font-size: 0.6875rem;
      line-height: 1.5;
      white-space: normal;
      width: 220px;
      max-width: 90vw;
      z-index: 99999;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.2s ease;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(137, 180, 250, 0.2);
      font-weight: 400;
      font-size: 0.6rem;
      text-transform: none;
      letter-spacing: 0;
    }

    /* Ensure tooltip doesn't get cut off on left side */
    th:first-child .tooltip,
    th:nth-child(2) .tooltip {
      left: 0;
      transform: translateX(0);
    }

    th:first-child .tooltip::after,
    th:nth-child(2) .tooltip::after {
      left: 1rem;
    }

    .info-icon:hover .tooltip {
      opacity: 1;
    }

    .tooltip::after {
      content: '';
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-bottom-color: rgba(30, 30, 46, 0.98);
    }

    .tooltip-desc {
      color: #bac2de;
    }

    /* Collapsible JSON */
    .json-line {
      display: flex;
      align-items: flex-start;
      font-family: 'Fira Code', 'SF Mono', Consolas, monospace;
    }

    .json-fold {
      cursor: pointer;
      user-select: none;
      color: #9399b2;
      margin-right: 0.125rem;
      width: 12px;
      flex-shrink: 0;
      text-align: center;
      font-size: 0.625rem;
      line-height: 1.5;
      transition: color 0.15s;
    }

    .json-fold:hover {
      color: var(--primary-color);
    }

    .json-fold.collapsed::before {
      content: '▶';
    }

    .json-fold.expanded::before {
      content: '▼';
    }

    .json-fold.empty {
      cursor: default;
      opacity: 0;
    }

    .json-content {
      flex: 1;
      min-width: 0;
    }

    .json-children {
      margin-left: 0.625rem;
    }

    .json-children.collapsed {
      display: none;
    }

    .json-ellipsis {
      color: #6c7086;
      font-style: italic;
    }

    /* Responsive styles using container queries */
    @container (max-width: 1200px) {
      .tree-panel {
        flex: 0 0 230px;
        min-width: 230px;
      }
    }

    @container (max-width: 900px) {
      .tree-panel {
        flex: 0 0 160px;
        min-width: 140px;
      }

      .tree-info-panel {
        height: 100px;
        min-height: 100px;
        max-height: 100px;
        font-size: 0.5625rem;
      }

      .tree-view {
        font-size: 0.625rem;
      }

      .code-block {
        font-size: 0.625rem;
      }

      /* Reduce table font size */
      .dimension-table {
        font-size: 0.6875rem;
      }

      .dimension-table input,
      .dimension-table select {
        font-size: 0.6875rem;
        padding: 0.2rem 0.25rem;
      }

      .dimension-table th {
        font-size: 0.5625rem;
        padding: 0.25rem 0.375rem;
      }

      /* Hide "Factor" suffix in column headers to save space */
      .hide-on-mobile {
        display: none;
      }

      /* Column widths determined by content at all breakpoints */
    }

    @container (max-width: 768px) {
      .tree-panel {
        flex: 0 0 160px;
        min-width: 160px;
      }

      .tree-info-panel {
        height: 150px;
        min-height: 80px;
        max-height: 170px;
        padding: 0.375rem 0.5rem;
        font-size: 0.5rem;
      }

      .tree-info-title {
        font-size: 0.5625rem;
      }

      .tree-view {
        font-size: 0.5625rem;
        padding: 0.375rem;
      }

      .tree-item {
        padding: 0.0625rem 0.125rem;
      }

      .code-area {
        min-height: 300px;
        max-height: 440px;
      }

      .tab-content {
        min-height: 220px;
      }

      /* Further reduce table font size */
      .dimension-table {
        font-size: 0.625rem;
      }

      .dimension-table input,
      .dimension-table select {
        font-size: 0.625rem;
        padding: 0.15rem 0.2rem;
      }

      .dimension-table th {
        font-size: 0.5rem;
        padding: 0.2rem 0.25rem;
        letter-spacing: 0;
      }

      .dimension-table td {
        padding: 0.15rem 0.25rem;
      }

      /* Column sizing by content continues */

      .toolbar {
        padding: 0.375rem 0.5rem;
        gap: 0.5rem;
      }

      .toolbar-group label {
        font-size: 0.5625rem;
      }

      .input-panel {
        padding: 0.5rem;
      }

      .add-dimension {
        font-size: 0.6875rem;
        padding: 0.3rem 0.4rem;
      }

      /* Info icon adjustments */
      .info-icon {
        width: 12px;
        height: 12px;
      }

      .tooltip {
        width: 180px;
        font-size: 0.5rem;
        padding: 0.375rem 0.5rem;
      }

      .tab {
        padding: 0.3rem 0.5rem;
        font-size: 0.625rem;
      }

      .version-toggle button {
        padding: 0.2rem 0.4rem;
        font-size: 0.5625rem;
      }

      button, select, input[type="number"], input[type="text"] {
        font-size: 0.6875rem;
        padding: 0.25rem 0.4rem;
      }

      .levels-input {
        width: 40px !important;
      }

      .preset-btn {
        padding: 0.25rem 0.4rem;
      }
    }

    @container (max-width: 600px) {
      .tree-panel {
        flex: 0 0 120px;
        min-width: 100px;
      }

      .tree-info-panel {
        display: none;
      }

      .code-block {
        font-size: 0.5rem;
        padding: 0.375rem;
      }

      .tree-view {
        font-size: 0.5rem;
      }

      /* Even smaller table */
      .dimension-table {
        font-size: 0.5625rem;
      }

      .dimension-table input,
      .dimension-table select {
        font-size: 0.5625rem;
        padding: 0.125rem 0.15rem;
      }

      .dimension-table th {
        font-size: 0.4375rem;
        padding: 0.15rem 0.2rem;
      }

      .dimension-table td {
        padding: 0.125rem 0.2rem;
      }

      /* Column sizing by content continues */
    }

    /* Plate Configuration Styles */
    .plate-section {
      margin-top: 0.75rem;
      border: 1px solid var(--border-color);
      border-radius: 4px;
      overflow: hidden;
    }

    .plate-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.625rem;
      background: rgba(64, 81, 181, 0.08);
      border-bottom: 1px solid var(--border-color);
      user-select: none;
    }

    .plate-header:hover {
      background: rgba(64, 81, 181, 0.12);
    }

    .plate-toggle {
      width: 16px;
      height: 16px;
      border: 2px solid var(--primary-color);
      border-radius: 3px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      background: var(--input-bg);
      transition: all 0.15s;
    }

    .plate-toggle.checked {
      background: var(--primary-color);
      color: white;
    }

    .plate-toggle.checked::after {
      content: '✓';
      font-size: 0.625rem;
      font-weight: bold;
    }

    .plate-title {
      flex: 1;
      font-size: 0.5625rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }

    .plate-collapse-icon {
      color: var(--text-muted);
      transition: transform 0.2s;
      font-size: 0.75rem;
      cursor: pointer;
      user-select: none;
      width: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .plate-collapse-icon.collapsed {
      transform: rotate(-90deg);
    }

    .plate-content {
      padding: 0.625rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      align-items: start;
    }

    .plate-content.collapsed {
      display: none;
    }

    .plate-controls {
      display: flex;
      flex-direction: column;
      gap: 0.625rem;
    }

    .plate-control-group {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .plate-control-group label {
      font-size: 0.6875rem;
      font-weight: 500;
      color: var(--text-muted);
    }

    .plate-type-select {
      width: 100%;
      padding: 0.25rem 0.4rem;
      border: 1px solid var(--border-color);
      border-radius: 4px;
      background: var(--input-bg);
      font-size: 0.6875rem;
      cursor: pointer;
    }

    .fov-input {
      width: 100%;
      padding: 0.25rem 0.4rem;
      border: 1px solid var(--border-color);
      border-radius: 4px;
      background: var(--input-bg);
      font-size: 0.6875rem;
    }

    /* Well Selector Styles */
    .well-selector {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem;
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.02);
      border-radius: 4px;
      max-width: 100%;
    }

    .well-selector-container {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .well-grid-wrapper {
      display: flex;
      gap: 0.25rem;
    }

    .well-row-label {
      width: 20px;
      min-width: 20px;
      text-align: right;
      font-size: 0.5625rem;
      color: var(--text-muted);
      font-weight: 500;
      display: flex;
      align-items: center;
      justify-content: flex-end;
    }

    .well-grid {
      display: grid;
      gap: 2px;
      background: var(--border-color);
      padding: 2px;
      border-radius: 3px;
      user-select: none;
      width: 300px;
      max-width: 100%;
    }

    .well-cell {
      aspect-ratio: 1 / 1;
      background: var(--input-bg);
      border: 1px solid var(--border-color);
      cursor: pointer;
      transition: all 0.1s;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.5rem;
      color: transparent;
    }

    .well-cell:hover {
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.1);
    }

    .well-cell.selected {
      background: var(--primary-color);
      color: white;
      border-color: var(--primary-color);
    }

    .well-cell.drag-range {
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.3);
      border-color: var(--primary-color);
    }

    .well-cell.selected::after {
      content: '✓';
      color: white;
    }

    .well-labels {
      display: grid;
      gap: 2px;
      font-size: 0.5625rem;
      color: var(--text-muted);
      font-weight: 500;
      padding-left: calc(20px + 0.25rem + 2px);
      padding-right: 2px;
      width: calc(300px + 20px + 0.25rem);
      max-width: 100%;
      margin-bottom: 0.25rem;
    }

    .well-label {
      text-align: center;
      min-width: 0;
    }

    .well-row-labels-container {
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 2px 0;
    }

    .well-actions {
      display: flex;
      gap: 0.375rem;
    }

    .well-action-btn {
      padding: 0.2rem 0.4rem;
      font-size: 0.625rem;
      border: 1px solid var(--border-color);
      background: var(--input-bg);
      border-radius: 3px;
      cursor: pointer;
      transition: all 0.15s;
    }

    .well-action-btn:hover {
      background: rgba(var(--md-primary-fg-color--rgb, 64, 81, 181), 0.08);
      border-color: var(--primary-color);
    }
  `;

  constructor() {
    super();
    this.version = 'v0.5';
    this.mode = 'image';
    this.activeTab = 'json';
    this.preset = '4d';
    this.levels = 2;
    this.numLevels = this.levels;
    this.copyButtonText = 'Copy';
    this.expandedNodes = { root: true, '0': false, '1': false, '2': false };
    this.selectedNode = 'root-meta';
    this.validationErrors = [];
    this.draggedIndex = null;
    this.collapsedJsonPaths = {};
    this.dimensions = [...(PRESETS[this.preset] || PRESETS['4d'])];
    this.validateDimensions();
    this._themeObserver = null;
    this._initializeTooltips();
    // Initialize plate properties
    this.plateControl = true;
    this.plateExpanded = false;
    this.plateEnabled = false;
    this.plateType = '24-well';
    this.selectedWells = [
      { row: 0, col: 0 }, // A1
      { row: 0, col: 1 }, // A2
      { row: 1, col: 0 }, // B1
      { row: 1, col: 1 }  // B2
    ];
    this.numFOVs = 1;
    // Drag selection state
    this.isDragging = false;
    this.dragStart = null;
    this.dragEnd = null;
  }

  connectedCallback() {
    super.connectedCallback();
    this._updateTheme();
    this._observeTheme();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._themeObserver) {
      this._themeObserver.disconnect();
    }
  }

  _updateTheme() {
    const scheme = document.body.getAttribute('data-md-color-scheme');
    this.setAttribute('data-theme', scheme || 'default');

    // Set CSS variables directly for Safari compatibility
    if (scheme === 'slate') {
      this.style.setProperty('--border-color', 'rgba(255, 255, 255, 0.08)');
      this.style.setProperty('--border-color-strong', 'rgba(255, 255, 255, 0.15)');
      this.style.setProperty('--text-muted', '#94a3b8');
      this.style.setProperty('--input-bg', 'hsl(232, 15%, 18%)');
      this.style.setProperty('--input-shadow', 'rgba(137, 180, 250, 0.2)');
    } else {
      this.style.setProperty('--border-color', 'rgba(0, 0, 0, 0.08)');
      this.style.setProperty('--border-color-strong', 'rgba(0, 0, 0, 0.15)');
      this.style.setProperty('--text-muted', '#64748b');
      this.style.setProperty('--input-bg', 'var(--md-default-bg-color, #fff)');
      this.style.setProperty('--input-shadow', 'rgba(64, 81, 181, 0.15)');
    }
  }

  _observeTheme() {
    this._themeObserver = new MutationObserver(() => this._updateTheme());
    this._themeObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme']
    });
  }

  _initializeTooltips() {
    this.tooltips = {
      name: {
        title: 'Name',
        desc: 'User-defined axis identifier. Can be anything, but common conventions: x/y/z for spatial, t for time, c for channel'
      },
      type: {
        title: 'Type',
        desc: 'Axis semantic type from the OME-NGFF spec. Options: space, time, channel'
      },
      unit: {
        title: 'Unit',
        desc: 'Physical unit for the axis (optional). Examples: micrometer, second, nanometer'
      },
      scale: {
        title: 'Scale',
        desc: 'Physical spacing per pixel at level 0. Example: 0.5 = 0.5 micrometers per pixel'
      },
      translation: {
        title: 'Translation',
        desc: 'Origin offset in physical coordinates. Used for positioning images with stage coordinates'
      },
      scaleFactor: {
        title: 'Downscale Factor',
        desc: 'Downsampling factor per pyramid level. Typically 2 for spatial axes, 1 for others'
      }
    };
  }

  // Valid units - delegates to ome_generator module
  getValidUnits(type) {
    return getValidUnits(type);
  }

  // Validate dimensions - delegates to ome_generator module
  validateDimensions() {
    const result = validateDimensions(this.dimensions);
    this.validationErrors = result.all;
  }

  // Plate methods
  togglePlate() {
    this.plateEnabled = !this.plateEnabled;

    // When toggling plate mode, automatically select the appropriate root metadata file
    // and expand the root node
    this.expandedNodes = { ...this.expandedNodes, root: true };

    if (this.plateEnabled && this.selectedWells.length > 0) {
      // Entering plate mode - select plate metadata
      this.selectedNode = 'root-plate-meta';
    } else {
      // Exiting plate mode - select image metadata
      this.selectedNode = 'root-meta';
    }
  }

  toggleWell(row, col) {
    const index = this.selectedWells.findIndex(
      w => w.row === row && w.col === col
    );

    if (index >= 0) {
      // Remove well
      this.selectedWells = this.selectedWells.filter((_, i) => i !== index);
    } else {
      // Add well
      this.selectedWells = [...this.selectedWells, { row, col }];
    }
  }

  selectAllWells() {
    const layout = this.getPlateLayout();
    const wells = [];
    for (let row = 0; row < layout.rows; row++) {
      for (let col = 0; col < layout.cols; col++) {
        wells.push({ row, col });
      }
    }
    this.selectedWells = wells;

    // If plate mode is enabled, select plate metadata
    if (this.plateEnabled) {
      this.expandedNodes = { ...this.expandedNodes, root: true };
      this.selectedNode = 'root-plate-meta';
    }
  }

  clearAllWells() {
    this.selectedWells = [];

    // If plate mode is enabled but no wells selected, fall back to image view
    if (this.plateEnabled) {
      this.selectedNode = 'root-meta';
    }
  }

  getPlateLayout() {
    const layouts = {
      '12-well': { rows: 3, cols: 4, rowNames: ['A', 'B', 'C'], colNames: ['1', '2', '3', '4'] },
      '24-well': { rows: 4, cols: 6, rowNames: ['A', 'B', 'C', 'D'], colNames: ['1', '2', '3', '4', '5', '6'] },
      '96-well': { rows: 8, cols: 12, rowNames: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], colNames: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'] },
    };
    return layouts[this.plateType] || layouts['96-well'];
  }

  isWellSelected(row, col) {
    return this.selectedWells.some(w => w.row === row && w.col === col);
  }

  // Drag selection methods
  startDragSelection(row, col) {
    this.isDragging = true;
    this.dragStart = { row, col };
    this.dragEnd = { row, col };
  }

  updateDragSelection(row, col) {
    if (!this.isDragging) return;
    this.dragEnd = { row, col };
    this.requestUpdate();
  }

  endDragSelection() {
    if (!this.isDragging) return;

    if (this.dragStart && this.dragEnd) {
      // Check if this was just a click (no drag)
      const isSingleClick = this.dragStart.row === this.dragEnd.row &&
                            this.dragStart.col === this.dragEnd.col;

      if (isSingleClick) {
        // Single click - just toggle the well
        this.toggleWell(this.dragStart.row, this.dragStart.col);
      } else {
        // Drag selection - select all wells in the dragged rectangle
        const minRow = Math.min(this.dragStart.row, this.dragEnd.row);
        const maxRow = Math.max(this.dragStart.row, this.dragEnd.row);
        const minCol = Math.min(this.dragStart.col, this.dragEnd.col);
        const maxCol = Math.max(this.dragStart.col, this.dragEnd.col);

        for (let r = minRow; r <= maxRow; r++) {
          for (let c = minCol; c <= maxCol; c++) {
            if (!this.isWellSelected(r, c)) {
              this.selectedWells = [...this.selectedWells, { row: r, col: c }];
            }
          }
        }
      }
    }

    this.isDragging = false;
    this.dragStart = null;
    this.dragEnd = null;
  }

  isWellInDragRange(row, col) {
    if (!this.isDragging || !this.dragStart || !this.dragEnd) return false;

    const minRow = Math.min(this.dragStart.row, this.dragEnd.row);
    const maxRow = Math.max(this.dragStart.row, this.dragEnd.row);
    const minCol = Math.min(this.dragStart.col, this.dragEnd.col);
    const maxCol = Math.max(this.dragStart.col, this.dragEnd.col);

    return row >= minRow && row <= maxRow && col >= minCol && col <= maxCol;
  }

  // Render a table header with an info icon tooltip
  // label can be a string or an object like {base: 'Text', hideOnMobile: ' Suffix'}
  renderHeaderWithTooltip(label, tooltipKey) {
    const tooltip = this.tooltips[tooltipKey];
    if (!tooltip) {
      return html`${label}`;
    }

    // Handle responsive labels
    let labelContent;
    if (typeof label === 'object' && label.base && label.hideOnMobile) {
      labelContent = html`${label.base}<span class="hide-on-mobile">${label.hideOnMobile}</span>`;
    } else {
      labelContent = label;
    }

    return html`
      <span class="header-with-tooltip">
        <span>${labelContent}</span>
        <span class="info-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 16v-4"/>
            <path d="M12 8h.01"/>
          </svg>
          <span class="tooltip">
            <div class="tooltip-desc">${tooltip.desc}</div>
          </span>
        </span>
      </span>
    `;
  }

  // Render well selector UI
  renderWellSelector() {
    const layout = this.getPlateLayout();

    return html`
      <div class="well-selector">
        <!-- Column labels -->
        <div class="well-labels" style="grid-template-columns: repeat(${layout.cols}, 1fr);">
          ${layout.colNames.map(name => html`<div class="well-label">${name}</div>`)}
        </div>

        <!-- Grid with row labels -->
        <div style="display: flex; gap: 0.25rem;">
          <!-- Row labels column -->
          <div class="well-row-labels-container">
            ${Array.from({ length: layout.rows }, (_, row) => html`
              <div class="well-row-label" style="flex: 1; display: flex; align-items: center; justify-content: flex-end;">
                ${layout.rowNames[row]}
              </div>
            `)}
          </div>

          <!-- Well grid -->
          <div
            class="well-grid"
            style="grid-template-columns: repeat(${layout.cols}, 1fr);"
            @mouseleave=${() => this.endDragSelection()}
            @mouseup=${() => this.endDragSelection()}
          >
            ${Array.from({ length: layout.rows }, (_, r) =>
              Array.from({ length: layout.cols }, (_, c) => {
                const selected = this.isWellSelected(r, c);
                const inDragRange = this.isWellInDragRange(r, c);
                return html`
                  <div
                    class="well-cell ${selected ? 'selected' : ''} ${inDragRange ? 'drag-range' : ''}"
                    @mousedown=${() => this.startDragSelection(r, c)}
                    @mouseenter=${() => this.updateDragSelection(r, c)}
                  ></div>
                `;
              })
            )}
          </div>
        </div>

        <!-- Well selector actions -->
        <div class="well-actions">
          <button class="well-action-btn" @click=${this.selectAllWells}>Select All</button>
          <button class="well-action-btn" @click=${this.clearAllWells}>Clear All</button>
          <span style="font-size: 0.6875rem; color: var(--text-muted);">
            ${this.selectedWells.length} well${this.selectedWells.length !== 1 ? 's' : ''} selected
          </span>
        </div>
      </div>
    `;
  }

  // Toggle JSON path collapsed state
  toggleJsonPath(path) {
    this.collapsedJsonPaths = {
      ...this.collapsedJsonPaths,
      [path]: !this.collapsedJsonPaths[path]
    };
  }

  // Render collapsible JSON
  renderCollapsibleJSON(obj, path = '', indent = 0, trailingComma = '') {
    if (obj === null) return html`<span class="syn-constant">null${trailingComma}</span>`;
    if (typeof obj === 'boolean') return html`<span class="syn-constant">${obj}${trailingComma}</span>`;
    if (typeof obj === 'number') return html`<span class="syn-number">${obj}${trailingComma}</span>`;
    if (typeof obj === 'string') return html`<span class="syn-string">"${obj}"${trailingComma}</span>`;

    const isCollapsed = this.collapsedJsonPaths[path];

    if (Array.isArray(obj)) {
      const isEmpty = obj.length === 0;
      if (isEmpty) {
        return html`[]${trailingComma}`;
      }

      // Check if it's a simple array (primitives only)
      const isSimple = obj.every(item =>
        typeof item === 'number' ||
        typeof item === 'string' ||
        typeof item === 'boolean' ||
        item === null
      );

      if (isSimple) {
        const content = obj.map((item, i) => {
          const comma = i < obj.length - 1 ? ', ' : '';
          if (typeof item === 'string') return `"${item}"${comma}`;
          if (typeof item === 'number') return html`<span class="syn-number">${item}${comma}</span>`;
          if (typeof item === 'boolean' || item === null) return html`<span class="syn-constant">${item}${comma}</span>`;
          return item;
        });
        return html`[${content}]${trailingComma}`;
      }

      return html`
        <div>
          <div class="json-line">
            <span
              class="json-fold ${isCollapsed ? 'collapsed' : 'expanded'}"
              @click=${() => this.toggleJsonPath(path)}
            ></span>
            <span class="json-content">
              [${isCollapsed ? html`<span class="json-ellipsis"> /* ${obj.length} items */ </span>]${trailingComma}` : ''}
            </span>
          </div>
          ${!isCollapsed ? html`
            <div class="json-children">
              ${obj.map((item, i) => {
                const comma = i < obj.length - 1 ? ',' : '';
                const itemContent = this.renderCollapsibleJSON(item, `${path}[${i}]`, indent + 1, comma);

                // Check if item is a primitive (single-line)
                const isPrimitive = typeof item !== 'object' || item === null ||
                  (Array.isArray(item) && item.every(x => typeof x !== 'object' || x === null)) ||
                  (typeof item === 'object' && Object.keys(item).length === 0);

                // Check if item is a simple object (all values are primitives)
                const isSimpleObject = typeof item === 'object' && item !== null && !Array.isArray(item) &&
                  Object.keys(item).length > 0 &&
                  Object.keys(item).every(key => {
                    const value = item[key];
                    return (
                      typeof value === 'number' ||
                      typeof value === 'string' ||
                      typeof value === 'boolean' ||
                      value === null
                    );
                  });

                // Render primitives and simple objects on their own line
                if (isPrimitive || isSimpleObject) {
                  return html`
                    <div class="json-line">
                      <span class="json-fold empty"></span>
                      <span class="json-content">${itemContent}</span>
                    </div>
                  `;
                } else {
                  // For complex items, just render them directly and append comma to closing bracket
                  return html`${itemContent}`;
                }
              })}
            </div>
            <div class="json-line">
              <span class="json-fold empty"></span>
              <span class="json-content">]${trailingComma}</span>
            </div>
          ` : ''}
        </div>
      `;
    }

    if (typeof obj === 'object') {
      const keys = Object.keys(obj);
      const isEmpty = keys.length === 0;

      if (isEmpty) {
        return html`{}${trailingComma}`;
      }

      return html`
        <div>
          <div class="json-line">
            <span
              class="json-fold ${isCollapsed ? 'collapsed' : 'expanded'}"
              @click=${() => this.toggleJsonPath(path)}
            ></span>
            <span class="json-content">
              {${isCollapsed ? html`<span class="json-ellipsis"> /* ${keys.length} ${keys.length === 1 ? 'property' : 'properties'} */ </span>}${trailingComma}` : ''}
            </span>
          </div>
          ${!isCollapsed ? html`
            <div class="json-children">
              ${keys.map((key, i) => {
                const value = obj[key];
                const valuePath = path ? `${path}.${key}` : key;
                const comma = i < keys.length - 1 ? ',' : '';
                const valueContent = this.renderCollapsibleJSON(value, valuePath, indent + 1, comma);

                // Check if value is a primitive (single-line)
                const isPrimitive = typeof value !== 'object' || value === null ||
                  (Array.isArray(value) && value.every(x => typeof x !== 'object' || x === null)) ||
                  (typeof value === 'object' && Object.keys(value).length === 0);

                return html`
                  <div class="json-line">
                    <span class="json-fold empty"></span>
                    <span class="json-content">
                      <span class="syn-property">"${key}"</span>: ${isPrimitive ? valueContent : html`<div style="display: inline-block; vertical-align: top; width: 100%;">${valueContent}</div>`}
                    </span>
                  </div>
                `;
              })}
            </div>
            <div class="json-line">
              <span class="json-fold empty"></span>
              <span class="json-content">}${trailingComma}</span>
            </div>
          ` : ''}
        </div>
      `;
    }

    return html`${String(obj)}${trailingComma}`;
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
    this.dimensions = [...PRESETS[preset]];
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
      const computedStyle = window.getComputedStyle(e.currentTarget);
    } else {
      e.currentTarget.classList.add('drag-placeholder-bottom');
      this.dropPosition = 'after';
      const computedStyle = window.getComputedStyle(e.currentTarget);
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

  // Generate JSON - delegates to ome_generator module
  generateJSON() {
    return generateJSON({
      dimensions: this.dimensions,
      version: this.version,
      numLevels: this.numLevels
    });
  }

  // Compact JSON formatter - delegates to ome_generator module
  compactJSON(obj, indent = 0) {
    return compactJSON(obj, indent);
  }

  // Generate tree data structure for ZarrTreeViewer
  generateTreeData() {
    const isV05 = this.version === 'v0.5';
    const metaFile = isV05 ? 'zarr.json' : '.zattrs';
    const arrayMeta = isV05 ? 'zarr.json' : '.zarray';
    const chunkPath = this.dimensions.map(d => d.name).join('/');

    // Helper: generate pyramid levels for an image
    const generateLevelNodes = (prefix = '') => {
      const levelNodes = [];
      for (let i = 0; i < this.numLevels; i++) {
        levelNodes.push({
          id: `${prefix}level-${i}`,
          label: `${i}/`,
          icon: 'folder',
          children: [
            {
              id: `${prefix}level-${i}-meta`,
              label: arrayMeta,
              icon: 'file',
            },
            {
              id: `${prefix}level-${i}-chunks`,
              label: chunkPath + '/...',
              icon: 'chunk',
            },
          ],
        });
      }
      return levelNodes;
    };

    // If plate mode is enabled, generate plate structure
    if (this.plateEnabled && this.selectedWells.length > 0) {
      const layout = this.getPlateLayout();
      const rootChildren = [
        {
          id: 'root-plate-meta',
          label: metaFile,
          icon: 'file',
        },
      ];

      // Create well folder structure
      const wellsByRow = {};
      this.selectedWells.forEach(({ row, col }) => {
        const rowName = layout.rowNames[row];
        if (!wellsByRow[rowName]) {
          wellsByRow[rowName] = [];
        }
        wellsByRow[rowName].push({ row, col, colName: layout.colNames[col] });
      });

      // Build row folders
      Object.keys(wellsByRow).sort().forEach(rowName => {
        const wells = wellsByRow[rowName];
        const rowChildren = [];

        wells.forEach(({ row, col, colName }) => {
          // FOV children
          const fovChildren = [];
          for (let f = 0; f < this.numFOVs; f++) {
            const fovPrefix = `well-${row}-${col}-fov-${f}-`;
            fovChildren.push({
              id: `well-${row}-${col}-fov-${f}`,
              label: `${f}/`,
              icon: 'folder',
              children: [
                {
                  id: `${fovPrefix}meta`,
                  label: metaFile,
                  icon: 'file',
                },
                ...generateLevelNodes(fovPrefix),
              ],
            });
          }

          // Well node
          rowChildren.push({
            id: `well-${row}-${col}`,
            label: `${colName}/`,
            icon: 'folder',
            children: [
              {
                id: `well-${row}-${col}-meta`,
                label: metaFile,
                icon: 'file',
              },
              ...fovChildren,
            ],
          });
        });

        // Row folder
        rootChildren.push({
          id: `row-${rowName}`,
          label: `${rowName}/`,
          icon: 'folder',
          children: rowChildren,
        });
      });

      return {
        id: 'root',
        label: 'plate.zarr/',
        icon: 'folder',
        children: rootChildren,
      };
    }

    // Otherwise, generate standard image structure
    const levelNodes = generateLevelNodes();

    const rootChildren = [
      {
        id: 'root-meta',
        label: metaFile,
        icon: 'file',
      },
    ];

    if (!isV05) {
      rootChildren.push({
        id: 'root-zgroup',
        label: '.zgroup',
        icon: 'file',
      });
    }

    rootChildren.push(...levelNodes);

    return {
      id: 'root',
      label: 'image.zarr/',
      icon: 'folder',
      children: rootChildren,
    };
  }

  // Generate file contents for ZarrTreeViewer
  generateFileContents() {
    const isV05 = this.version === 'v0.5';
    const metaFile = isV05 ? 'zarr.json' : '.zattrs';
    const arrayMeta = isV05 ? 'zarr.json' : '.zarray';
    const contents = {};

    // Helper: generate array level contents
    const generateArrayLevels = (prefix = '') => {
      for (let i = 0; i < this.numLevels; i++) {
        const isFirst = i === 0;
        const levelDesc = isFirst ? 'Full resolution' : `downsampled level ${i}`;

        contents[`${prefix}level-${i}`] = {
          title: `${i}/`,
          description: `${levelDesc} pyramid level. Contains the array data as chunked storage.`,
          content: null,
        };

        contents[`${prefix}level-${i}-meta`] = {
          title: arrayMeta,
          description: isV05
            ? `Zarr v3 array metadata. Defines shape, chunks, dtype, codecs, and dimension_names matching the axes: [${this.dimensions.map(d => `"${d.name}"`).join(', ')}].`
            : `Zarr v2 array metadata. Defines shape, chunks, dtype, compressor, and dimension_separator.`,
          content: JSON.parse(this.generateArrayMetadataJSON(i)),
        };

        contents[`${prefix}level-${i}-chunks`] = {
          title: this.dimensions.map(d => d.name).join('/') + '/...',
          description: `Chunk files organized by dimension. Each chunk contains a portion of the array data compressed according to the codec settings.`,
          content: null,
        };
      }
    };

    // If plate mode is enabled
    if (this.plateEnabled && this.selectedWells.length > 0) {
      const layout = this.getPlateLayout();

      contents['root'] = {
        title: 'plate.zarr/',
        description: `Root group of the OME-Zarr plate. Contains ${isV05 ? 'zarr.json with OME plate metadata' : 'plate metadata in .zattrs'}.`,
        content: null,
      };

      contents['root-plate-meta'] = {
        title: metaFile,
        description: isV05
          ? 'Zarr v3 group metadata with plate layout. Defines rows, columns, and wells with their positions.'
          : 'Plate metadata file defining the well plate structure.',
        content: JSON.parse(generatePlateJSON({
          version: this.version,
          plateType: this.plateType,
          selectedWells: this.selectedWells,
          numFOVs: this.numFOVs
        })),
      };

      // Generate well metadata
      this.selectedWells.forEach(({ row, col }) => {
        const rowName = layout.rowNames[row];
        const colName = layout.colNames[col];

        contents[`row-${rowName}`] = {
          title: `${rowName}/`,
          description: `Row ${rowName} of the plate.`,
          content: null,
        };

        contents[`well-${row}-${col}`] = {
          title: `${colName}/`,
          description: `Well ${rowName}${colName}. Contains ${this.numFOVs} field(s) of view.`,
          content: null,
        };

        contents[`well-${row}-${col}-meta`] = {
          title: metaFile,
          description: `Well metadata listing all fields of view (FOVs) for this well.`,
          content: JSON.parse(generateWellJSON({
            version: this.version,
            numFOVs: this.numFOVs
          })),
        };

        // FOV metadata
        for (let f = 0; f < this.numFOVs; f++) {
          const fovPrefix = `well-${row}-${col}-fov-${f}-`;

          contents[`well-${row}-${col}-fov-${f}`] = {
            title: `${f}/`,
            description: `Field of view ${f}. Contains multiscale image pyramid.`,
            content: null,
          };

          contents[`${fovPrefix}meta`] = {
            title: metaFile,
            description: `Image metadata for FOV ${f}. Contains multiscales with axes and transformations.`,
            content: JSON.parse(this.generateJSON()),
          };

          generateArrayLevels(fovPrefix);
        }
      });

      return contents;
    }

    // Otherwise, generate standard image structure
    contents['root'] = {
      title: 'image.zarr/',
      description: `Root group of the OME-Zarr image. Contains ${isV05 ? 'zarr.json with OME metadata under attributes.ome' : '.zattrs with multiscales metadata'}.`,
      content: null,
    };

    contents['root-meta'] = {
      title: metaFile,
      description: isV05
        ? 'Zarr v3 group metadata file. Contains zarr_format, node_type, and OME metadata under attributes.ome.multiscales.'
        : 'Zarr v2 attributes file containing the multiscales array with axes and coordinate transformations.',
      content: JSON.parse(this.generateJSON()),
    };

    if (!isV05) {
      contents['root-zgroup'] = {
        title: '.zgroup',
        description: 'Zarr v2 group marker file. Contains {"zarr_format": 2} to identify this as a Zarr group.',
        content: { zarr_format: 2 },
      };
    }

    generateArrayLevels();

    return contents;
  }

  // Generate array metadata JSON - delegates to ome_generator module
  generateArrayMetadataJSON(level) {
    return generateArrayMetadataJSON({
      dimensions: this.dimensions,
      version: this.version
    }, level);
  }

  // Generate Python code - delegates to ome_generator module
  generatePython() {
    if (this.plateEnabled && this.selectedWells.length > 0) {
      return generatePlatePython({
        dimensions: this.dimensions,
        version: this.version,
        numLevels: this.numLevels,
        plateType: this.plateType,
        selectedWells: this.selectedWells,
        numFOVs: this.numFOVs
      });
    }

    return generatePython({
      dimensions: this.dimensions,
      version: this.version,
      numLevels: this.numLevels
    });
  }

  async copyJSONToClipboard() {
    const jsonString = this.generateJSONForSelectedNode();
    if (!jsonString) return;

    await navigator.clipboard.writeText(jsonString);

    // Visual feedback
    this.copyButtonText = 'Copied!';
    this.requestUpdate();
    setTimeout(() => {
      this.copyButtonText = 'Copy';
      this.requestUpdate();
    }, 2000);
  }

  async copyToClipboard() {
    // For Python tab
    const text = this.generatePython();
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
      const jsonString = this.generateJSONForSelectedNode();
      if (jsonString === null) {
        return html`<span class="syn-comment">// Select a metadata file (.zattrs, .zarray, .zgroup, or zarr.json)<br/>// to see its contents</span>`;
      }
      // Parse and render collapsible JSON
      try {
        const jsonObj = JSON.parse(jsonString);
        return this.renderCollapsibleJSON(jsonObj);
      } catch (e) {
        return html`<span class="syn-comment">// Error parsing JSON: ${e.message}</span>`;
      }
    } else {
      return unsafeHTML(this.highlightPython(this.generatePython()));
    }
  }

  // Returns true if the selected node is a metadata file
  isMetadataFile(nodeId) {
    if (!nodeId) return false;
    // root-meta is .zattrs (v2) or zarr.json (v3) for the root group
    // root-zgroup is .zgroup (v2 only)
    // root-plate-meta is zarr.json for plate
    // well-X-Y-meta is zarr.json for well
    // well-X-Y-fov-Z-meta is zarr.json for FOV image
    // level-N-meta is .zarray (v2) or zarr.json (v3) for arrays
    // *-zgroup is .zgroup (v2 only) for any group
    return nodeId === 'root-meta' ||
           nodeId === 'root-zgroup' ||
           nodeId === 'root-plate-meta' ||
           nodeId.endsWith('-meta') ||
           nodeId.endsWith('-zgroup');
  }

  generateJSONForSelectedNode() {
    const nodeId = this.selectedNode;
    if (!this.isMetadataFile(nodeId)) {
      return null;
    }

    // Plate metadata (root-plate-meta)
    if (nodeId === 'root-plate-meta') {
      return generatePlateJSON({
        version: this.version,
        plateType: this.plateType,
        selectedWells: this.selectedWells,
        numFOVs: this.numFOVs
      });
    }

    // Well metadata (well-X-Y-meta)
    if (nodeId.startsWith('well-') && nodeId.endsWith('-meta')) {
      const parts = nodeId.split('-');
      // Check if it's a well metadata (not FOV metadata)
      // well-X-Y-meta has 4 parts, well-X-Y-fov-Z-meta has 6 parts
      if (parts.length === 4) {
        return generateWellJSON({
          version: this.version,
          numFOVs: this.numFOVs
        });
      }
      // FOV metadata (well-X-Y-fov-Z-meta) - 6 parts
      if (parts.length === 6 && parts[3] === 'fov') {
        return this.generateJSON();
      }
      // FOV level metadata (well-X-Y-fov-Z-level-N-meta)
      if (parts.length === 8 && parts[3] === 'fov' && parts[5] === 'level') {
        const level = parseInt(parts[6]);
        return this.generateArrayJSON(level);
      }
    }

    if (nodeId === 'root-meta') {
      // Root group metadata (.zattrs or zarr.json)
      return this.generateJSON();
    }

    if (nodeId === 'root-zgroup' || nodeId.endsWith('-zgroup')) {
      // .zgroup file (v2 only) - all groups have the same content
      return this.compactJSON({ zarr_format: 2 });
    }

    // Array metadata (level-N-meta) for non-plate image mode
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

    // Check if we're in plate mode
    const isPlateMode = this.plateEnabled && this.selectedWells.length > 0;

    const infos = {
      'root': {
        title: isPlateMode ? 'plate.zarr/' : 'image.zarr/',
        desc: isPlateMode
          ? `Root group of the OME-Zarr plate. Contains ${isV05 ? 'zarr.json with OME plate metadata' : 'plate metadata in .zattrs'}.`
          : `Root group of the OME-Zarr image. Contains ${isV05 ? 'zarr.json with OME metadata under attributes.ome' : '.zattrs with multiscales metadata'}.`
      },
      'root-meta': {
        title: metaFile,
        desc: isPlateMode
          ? (isV05
              ? 'Zarr v3 group metadata with plate layout. Defines rows, columns, and wells with their positions.'
              : 'Plate metadata file defining the well plate structure.')
          : (isV05
              ? 'Zarr v3 group metadata file. Contains zarr_format, node_type, and OME metadata under attributes.ome.multiscales.'
              : 'Zarr v2 attributes file containing the multiscales array with axes and coordinate transformations.')
      },
      'root-plate-meta': {
        title: metaFile,
        desc: isV05
          ? 'Zarr v3 group metadata with plate layout. Defines rows, columns, and wells with their positions.'
          : 'Plate metadata file defining the well plate structure.'
      },
      'root-zgroup': {
        title: '.zgroup',
        desc: 'Zarr v2 group marker file. Contains {"zarr_format": 2} to identify this as a Zarr group.'
      },
      'root-plate-zgroup': {
        title: '.zgroup',
        desc: 'Zarr v2 group marker file. Contains {"zarr_format": 2} to identify this as a Zarr group.'
      },
    };

    // Handle .zgroup files for any group (plate, row, well, FOV, or image)
    if (nodeId.endsWith('-zgroup')) {
      return {
        title: '.zgroup',
        desc: 'Zarr v2 group marker file. Contains {"zarr_format": 2} to identify this as a Zarr group.'
      };
    }

    // Handle plate-specific nodes (row folders, wells, FOVs)
    if (isPlateMode && nodeId.startsWith('row-')) {
      const rowName = nodeId.substring(4);
      return {
        title: `${rowName}/`,
        desc: `Row ${rowName} of the plate.`
      };
    }

    if (nodeId.startsWith('well-')) {
      const parts = nodeId.split('-');
      if (parts.length >= 3) {
        const row = parseInt(parts[1]);
        const col = parseInt(parts[2]);
        const layout = this.getPlateLayout();
        const rowName = layout.rowNames[row];
        const colName = layout.colNames[col];

        if (nodeId.endsWith('-meta')) {
          return {
            title: metaFile,
            desc: `Well metadata listing all fields of view (FOVs) for well ${rowName}${colName}.`
          };
        } else if (parts.length >= 5 && parts[3] === 'fov') {
          const fovNum = parts[4];
          const prefix = `well-${row}-${col}-fov-${fovNum}-`;

          if (nodeId === `well-${row}-${col}-fov-${fovNum}`) {
            return {
              title: `${fovNum}/`,
              desc: `Field of view ${fovNum} in well ${rowName}${colName}. Contains multiscale image pyramid.`
            };
          } else if (nodeId === `${prefix}meta`) {
            return {
              title: metaFile,
              desc: `Image metadata for FOV ${fovNum}. Contains multiscales with axes and transformations.`
            };
          } else if (nodeId.includes('-level-')) {
            // Handle level nodes
            const levelMatch = nodeId.match(/level-(\d+)(-meta|-chunks)?$/);
            if (levelMatch) {
              const levelNum = levelMatch[1];
              const suffix = levelMatch[2];
              const isFirst = levelNum === '0';
              const levelDesc = isFirst ? 'Full resolution pyramid level' : `downsampled pyramid level ${levelNum}`;

              if (!suffix) {
                return {
                  title: `${levelNum}/`,
                  desc: `${levelDesc}. Contains the array data as chunked storage.`
                };
              } else if (suffix === '-meta') {
                return {
                  title: arrayMeta,
                  desc: isV05
                    ? `Zarr v3 array metadata. Defines shape, chunks, dtype, codecs, and dimension_names matching the axes: [${this.dimensions.map(d => `"${d.name}"`).join(', ')}].`
                    : `Zarr v2 array metadata. Defines shape, chunks, dtype, compressor, and dimension_separator.`
                };
              } else if (suffix === '-chunks') {
                return {
                  title: this.dimensions.map(d => d.name).join('/') + '/...',
                  desc: `Chunk files organized by dimension. Each chunk contains a portion of the array data compressed according to the codec settings.`
                };
              }
            }
          }
        } else {
          return {
            title: `${colName}/`,
            desc: `Well ${rowName}${colName}. Contains ${this.numFOVs} field(s) of view.`
          };
        }
      }
    }

    // Add level info dynamically for image mode
    for (let i = 0; i < this.numLevels; i++) {
      const isFirst = i === 0;
      const levelDesc = isFirst ? 'Full resolution pyramid level' : `downsampled pyramid level ${i}`;
      infos[`level-${i}`] = {
        title: `${i}/`,
        desc: `${levelDesc}. Contains the array data as chunked storage.`
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

    // Helper: generate level nodes for a given prefix
    const generateLevelNodes = (prefix = '') => {
      const levelNodes = [];
      for (let i = 0; i < this.numLevels; i++) {
        levelNodes.push(
          this.renderTreeNode(`${prefix}level-${i}`, `${i}/`, 'folder', true, html`
            ${this.renderTreeNode(`${prefix}level-${i}-meta`, arrayMeta, 'file', false)}
            ${this.renderTreeNode(`${prefix}level-${i}-chunks`, chunkPath + '/...', 'chunk', false)}
          `)
        );
      }
      return levelNodes;
    };

    // If plate mode is enabled, render plate structure
    if (this.plateEnabled && this.selectedWells.length > 0) {
      const layout = this.getPlateLayout();

      // Group wells by row
      const wellsByRow = {};
      this.selectedWells.forEach(({ row, col }) => {
        const rowName = layout.rowNames[row];
        if (!wellsByRow[rowName]) {
          wellsByRow[rowName] = [];
        }
        wellsByRow[rowName].push({ row, col, colName: layout.colNames[col] });
      });

      // Sort wells within each row by column
      Object.keys(wellsByRow).forEach(rowName => {
        wellsByRow[rowName].sort((a, b) => a.col - b.col);
      });

      // Generate row folders
      const rowFolders = Object.keys(wellsByRow).sort().map(rowName => {
        const wells = wellsByRow[rowName];

        // Generate well folders for this row
        const wellFolders = wells.map(({ row, col, colName }) => {
          // Generate FOV folders for this well
          const fovFolders = [];
          for (let f = 0; f < this.numFOVs; f++) {
            const fovPrefix = `well-${row}-${col}-fov-${f}-`;
            fovFolders.push(
              this.renderTreeNode(`well-${row}-${col}-fov-${f}`, `${f}/`, 'folder', true, html`
                ${this.renderTreeNode(`${fovPrefix}meta`, metaFile, 'file', false)}
                ${!isV05 ? this.renderTreeNode(`${fovPrefix}zgroup`, '.zgroup', 'file', false) : ''}
                ${generateLevelNodes(fovPrefix)}
              `)
            );
          }

          return this.renderTreeNode(`well-${row}-${col}`, `${colName}/`, 'folder', true, html`
            ${this.renderTreeNode(`well-${row}-${col}-meta`, metaFile, 'file', false)}
            ${!isV05 ? this.renderTreeNode(`well-${row}-${col}-zgroup`, '.zgroup', 'file', false) : ''}
            ${fovFolders}
          `);
        });

        return this.renderTreeNode(`row-${rowName}`, `${rowName}/`, 'folder', true, html`
          ${!isV05 ? this.renderTreeNode(`row-${rowName}-zgroup`, '.zgroup', 'file', false) : ''}
          ${wellFolders}
        `);
      });

      return html`
        <div class="tree-view">
          ${this.renderTreeNode('root', 'plate.zarr/', 'folder', true, html`
            ${this.renderTreeNode('root-plate-meta', metaFile, 'file', false)}
            ${!isV05 ? this.renderTreeNode('root-plate-zgroup', '.zgroup', 'file', false) : ''}
            ${rowFolders}
          `)}
        </div>
        <div class="tree-info-panel">
          ${this.selectedNode ? html`
            <div class="tree-info-title">${this.getNodeInfo(this.selectedNode).title}</div>
            <div class="tree-info-desc">${this.getNodeInfo(this.selectedNode).desc}</div>
          ` : html`
            <div class="tree-info-hint">Click a file or folder to see its purpose</div>
          `}
        </div>
      `;
    }

    // Otherwise, render standard image structure
    const levelNodes = generateLevelNodes();

    return html`
      <div class="tree-view">
        ${this.renderTreeNode('root', 'image.zarr/', 'folder', true, html`
          ${this.renderTreeNode('root-meta', metaFile, 'file', false)}
          ${!isV05 ? this.renderTreeNode('root-zgroup', '.zgroup', 'file', false) : ''}
          ${levelNodes}
        `)}
      </div>
      <div class="tree-info-panel">
        ${this.selectedNode ? html`
          <div class="tree-info-title">${this.getNodeInfo(this.selectedNode).title}</div>
          <div class="tree-info-desc">${this.getNodeInfo(this.selectedNode).desc}</div>
        ` : html`
          <div class="tree-info-hint">Click a file or folder to see its purpose</div>
        `}
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
                    <th>${this.renderHeaderWithTooltip('Name', 'name')}</th>
                    <th>${this.renderHeaderWithTooltip('Type', 'type')}</th>
                    <th>${this.renderHeaderWithTooltip('Unit', 'unit')}</th>
                    <th>${this.renderHeaderWithTooltip('Scale', 'scale')}</th>
                    <th>${this.renderHeaderWithTooltip({base: 'Trans', hideOnMobile: 'late'}, 'translation')}</th>
                    <th>${this.renderHeaderWithTooltip({base: 'Downscale', hideOnMobile: ' Factor'}, 'scaleFactor')}</th>
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

            <!-- Plate Configuration Section -->
            ${this.plateControl ? html`
            <div class="plate-section">
              <div class="plate-header">
                <div class="plate-collapse-icon ${!this.plateExpanded ? 'collapsed' : ''}" @click=${() => this.plateExpanded = !this.plateExpanded}>▼</div>
                <div
                  class="plate-toggle ${this.plateEnabled ? 'checked' : ''}"
                  @click=${() => this.togglePlate()}
                ></div>
                <div class="plate-title" @click=${() => this.togglePlate()}>Plate Configuration</div>
              </div>

              <div class="plate-content ${!this.plateExpanded ? 'collapsed' : ''}">
                <!-- Plate Controls -->
                <div class="plate-controls">
                  <div class="plate-control-group">
                    <label>Plate Type:</label>
                    <select
                      class="plate-type-select"
                      .value=${this.plateType}
                      @change=${(e) => this.plateType = e.target.value}
                    >
                      <option value="12-well">12-well (3×4)</option>
                      <option value="24-well">24-well (4×6)</option>
                      <option value="96-well">96-well (8×12)</option>
                    </select>
                  </div>

                  <div class="plate-control-group">
                    <label>FOVs per well:</label>
                    <input
                      type="number"
                      class="fov-input"
                      min="1"
                      max="5"
                      .value=${this.numFOVs}
                      @input=${(e) => this.numFOVs = parseInt(e.target.value) || 1}
                    />
                  </div>
                </div>

                <!-- Well Selector -->
                ${this.renderWellSelector()}
              </div>
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
              <!-- Tree panel - always visible -->
              <div class="tree-panel">
                ${this.renderTree()}
              </div>

              <!-- Content area - switches between JSON and Python -->
              <div class="code-output">
                ${this.activeTab === 'json' ? html`
                  <div class="tab-content">
                    <button
                      class="copy-button ${this.copyButtonText.includes('Copied') ? 'copied' : ''}"
                      @click=${() => this.copyJSONToClipboard()}
                    >
                      ${this.copyButtonText.includes('Copied') ? '✓' : '📋'} ${this.copyButtonText}
                    </button>
                    <div class="code-block">
                      ${this.getHighlightedCode()}
                    </div>
                  </div>
                ` : html`
                  <div class="tab-content">
                    <button
                      class="copy-button ${this.copyButtonText.includes('Copied') ? 'copied' : ''}"
                      @click=${this.copyToClipboard}
                    >
                      ${this.copyButtonText.includes('Copied') ? '✓' : '📋'} ${this.copyButtonText}
                    </button>
                    <div class="code-block"><pre>${unsafeHTML(this.highlightPython(this.generatePython()))}</pre></div>
                  </div>
                `}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }
}

customElements.define('ome-explorer', OmeExplorer);
