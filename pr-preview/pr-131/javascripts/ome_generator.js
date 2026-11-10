/**
 * OME-Zarr Metadata Generator
 *
 * Pure functions for generating OME-NGFF metadata JSON and Python code.
 * This module has no DOM dependencies and can be used in both browser and Node.js.
 *
 * The web component (ome_explorer.js) imports these functions.
 * Tests (tests/ome_explorer_runner.js) also import these for validation.
 */

// Units are auto-generated from Python by scripts/export_units.py
import { units } from './ome_units.js';

/**
 * @typedef {Object} Dimension
 * @property {string} name - Dimension name (anything)
 * @property {'space'|'time'|'channel'} [type] - Axis type
 * @property {string} [unit] - Unit of measurement (e.g., 'micrometer', 'second')
 * @property {number} scale - Scale factor for this dimension
 * @property {number} [translation=0] - Translation offset
 * @property {number} [scaleFactor=1] - Scale factor per pyramid level
 */

/**
 * Get valid units for an axis type.
 * @param {string} type - 'space', 'time', or 'channel'
 * @returns {string[]} Array of valid unit names
 */
export function getValidUnits(type) {
  return units[type] || [];
}

/**
 * Shared logic: Check if an array should be compacted to one line.
 * @param {Array} arr - Array to check
 * @returns {boolean} True if array should be on one line
 */
export function shouldCompactArray(arr) {
  return arr.every(item =>
    typeof item === 'number' ||
    typeof item === 'string' ||
    typeof item === 'boolean' ||
    item === null
  );
}

/**
 * Custom JSON formatter that keeps simple arrays on single lines.
 * @param {any} obj - Object to format
 * @param {number} indent - Current indentation level
 * @returns {string} Formatted JSON string
 */
export function compactJSON(obj, indent = 0, maxLength = 79) {
  const spaces = '  '.repeat(indent);
  const nextSpaces = '  '.repeat(indent + 1);

  if (obj === null) return 'null';
  if (typeof obj === 'boolean') return obj ? 'true' : 'false';
  if (typeof obj === 'number') return String(obj);
  if (typeof obj === 'string') return JSON.stringify(obj);

  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';

    if (shouldCompactArray(obj)) {
      // Keep simple arrays on one line
      return '[' + obj.map(item =>
        typeof item === 'string' ? JSON.stringify(item) : String(item)
      ).join(', ') + ']';
    }

    // Complex arrays get one item per line
    const items = obj.map(item => nextSpaces + compactJSON(item, indent + 1, maxLength));
    return '[\n' + items.join(',\n') + '\n' + spaces + ']';
  }

  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (keys.length === 0) return '{}';

    // Always multi-line format for objects
    const items = keys.map(key => {
      const value = compactJSON(obj[key], indent + 1, maxLength);
      return nextSpaces + JSON.stringify(key) + ': ' + value;
    });
    return '{\n' + items.join(',\n') + '\n' + spaces + '}';
  }

  return String(obj);
}

/**
 * Generate OME-NGFF multiscales JSON metadata.
 *
 * @param {Object} config - Configuration object
 * @param {Dimension[]} config.dimensions - Array of dimension objects
 * @param {string} config.version - 'v0.4' or 'v0.5'
 * @param {number} config.numLevels - Number of pyramid levels
 * @returns {string} Formatted JSON string
 */
export function generateJSON({ dimensions, version, numLevels }) {
  const axes = dimensions.map(d => {
    const axis = { name: d.name, type: d.type || undefined };
    if (d.unit) axis.unit = d.unit;
    return axis;
  });

  const datasets = [];
  for (let level = 0; level < numLevels; level++) {
    const scale = dimensions.map(d =>
      d.scale * Math.pow(d.scaleFactor || 1, level)
    );
    const transforms = [{ scale }];

    if (dimensions.some(d => d.translation !== 0)) {
      transforms.push({
        translation: dimensions.map(d => d.translation || 0)
      });
    }

    datasets.push({
      path: String(level),
      coordinateTransformations: transforms
    });
  }

  const multiscale = version === 'v0.4'
    ? {
        version: '0.4',
        name: 'example_image',
        axes,
        datasets,
      }
    : {
        name: 'example_image',
        axes,
        datasets,
      };

  // if (version === 'v0.5') {
  //   // v0.5 MAY uses coordinateTransformations at multiscale level
  //   multiscale.coordinateTransformations = [{
  //     scale: [1, 1, 1, 1, 1].slice(0, dimensions.length)
  //   }];
  // }

  // v0.5: zarr.json with zarr_format, node_type, attributes
  // v0.4: .zattrs with just the multiscales array
  const output = version === 'v0.5'
    ? {
        zarr_format: 3,
        node_type: 'group',
        attributes: {
          ome: { version: "0.5", multiscales: [multiscale] }
        }
      }
    : { multiscales: [multiscale] };

  return compactJSON(output);
}

/**
 * Generate array metadata JSON for a specific pyramid level.
 *
 * @param {Object} config - Configuration object
 * @param {Dimension[]} config.dimensions - Array of dimension objects
 * @param {string} config.version - 'v0.4' or 'v0.5'
 * @param {number} level - Pyramid level (0 = full resolution)
 * @returns {string} Formatted JSON string
 */
export function generateArrayMetadataJSON({ dimensions, version }, level) {
  const isV05 = version === 'v0.5';
  const base = 512;
  const shape = dimensions.map(d => {
    if (d.type === 'space') {
      return Math.max(1, Math.floor(base / Math.pow(d.scaleFactor || 2, level)));
    }
    return base;
  });

  const chunks = dimensions.map(d => {
    if (d.type === 'space') return 64;
    if (d.type === 'channel') return 1;
    if (d.type === 'time') return 1;
    return 64;
  });

  if (isV05) {
    // Zarr v3 array metadata
    return compactJSON({
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
      dimension_names: dimensions.map(d => d.name)
    });
  } else {
    // Zarr v2 .zarray
    return compactJSON({
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

/**
 * Generate Python code using yaozarrs library.
 * Includes both explicit (Method 1) and DimSpec (Method 2) approaches.
 *
 * @param {Object} config - Configuration object
 * @param {Dimension[]} config.dimensions - Array of dimension objects
 * @param {string} config.version - 'v0.4' or 'v0.5'
 * @param {number} config.numLevels - Number of pyramid levels
 * @returns {string} Python code string
 */
export function generatePython({ dimensions, version, numLevels }) {
  const ver = version.replace('.', '');

  const axesCode = dimensions.map(d => {
    const typeMap = {
      'space': 'SpaceAxis',
      'time': 'TimeAxis',
      'channel': 'ChannelAxis',
    };
    const axisClass = typeMap[d.type] || 'CustomAxis';
    const unit = d.unit ? `, unit="${d.unit}"` : '';
    return `    ${ver}.${axisClass}(name="${d.name}"${unit})`;
  }).join(',\n');

  const dimsCode = dimensions.map(d => {
    const parts = [`name="${d.name}"`];
    if (d.type) parts.push(`type="${d.type}"`);
    if (d.unit) parts.push(`unit="${d.unit}"`);
    if (d.scale !== 1) parts.push(`scale=${d.scale}`);
    if (d.translation !== 0) parts.push(`translation=${d.translation}`);
    if (d.scaleFactor !== 1) parts.push(`scale_factor=${d.scaleFactor}`);
    return `    DimSpec(${parts.join(', ')})`;
  }).join(',\n');

  return `from yaozarrs import ${ver}, DimSpec

# ###############################################
# Method 1: Using Axis & Dataset classes directly
# ###############################################

axes = [
${axesCode}
]

datasets = [
${Array.from({ length: numLevels }, (_, level) => {
    const scales = dimensions.map(d =>
      d.scale * (d.scaleFactor || 1) ** level
    ).join(', ');
    const transforms = [`${ver}.ScaleTransformation(scale=[${scales}])`];

    if (dimensions.some(d => d.translation !== 0)) {
      const translations = dimensions.map(d => d.translation || 0).join(', ');
      transforms.push(`${ver}.TranslationTransformation(translation=[${translations}])`);
    }

    return `    ${ver}.Dataset(
        path="${level}",
        coordinateTransformations=[
            ${transforms.join(',\n            ')}
        ]
    )`;
  }).join(',\n')}
]

multiscale1 = ${ver}.Multiscale(
    name="example_image",
    axes=axes,
    datasets=datasets
)

# ###############################################
# Method 2: Using DimSpec
# ###############################################

dims = [
${dimsCode}
]

multiscale2 = ${ver}.Multiscale.from_dims(
    dims,
    name="example_image",
    n_levels=${numLevels}
)

# ###############################################
# Shared code: Construct Image and print JSON
# ###############################################

# Convert to JSON
image = ${ver}.Image(multiscales=[multiscale2])
print(image.model_dump_json(indent=2))`;
}

/**
 * Validate dimension configuration according to OME-NGFF spec.
 *
 * @param {Dimension[]} dimensions - Array of dimension objects
 * @returns {Object} Object with 'errors' and 'warnings' arrays
 */
export function validateDimensions(dimensions) {
  const errors = [];
  const warnings = [];

  // Count axes by type
  const typeCounts = {
    space: dimensions.filter(d => d.type === 'space').length,
    time: dimensions.filter(d => d.type === 'time').length,
    channel: dimensions.filter(d => d.type === 'channel').length,
  };

  // Rule 1: Must have 2-5 dimensions total
  if (dimensions.length < 2) {
    errors.push({
      type: 'error',
      message: `Too few dimensions (${dimensions.length}). OME-NGFF requires at least 2 dimensions.`,
      hint: 'Add more spatial dimensions (x, y, or z).'
    });
  }
  if (dimensions.length > 5) {
    errors.push({
      type: 'error',
      message: `Too many dimensions (${dimensions.length}). OME-NGFF allows maximum 5 dimensions.`,
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
  const actualOrder = dimensions.map(d => typeOrder[d.type] ?? 3);
  const sortedOrder = [...actualOrder].sort((a, b) => a - b);

  if (JSON.stringify(actualOrder) !== JSON.stringify(sortedOrder)) {
    errors.push({
      type: 'error',
      message: 'Dimensions are not in the required order.',
      hint: 'Order must be: [time,] [channel,] then space axes. Try: t, c, z, y, x'
    });
  }

  // Rule 6: Check for duplicate names
  const names = dimensions.map(d => d.name);
  const duplicates = names.filter((name, i) => names.indexOf(name) !== i);
  if (duplicates.length > 0) {
    errors.push({
      type: 'error',
      message: `Duplicate axis names: ${[...new Set(duplicates)].join(', ')}`,
      hint: 'Each dimension must have a unique name.'
    });
  }

  // Rule 7: Validate units (warnings, not errors)
  dimensions.forEach((dim) => {
    if (dim.unit) {
      const validUnits = units[dim.type] ?? [];
      if (validUnits.length > 0 && !validUnits.includes(dim.unit)) {
        warnings.push({
          type: 'warning',
          message: `Dimension "${dim.name}": unit "${dim.unit}" is not a recognized ${dim.type} unit.`,
          hint: `Valid ${dim.type} units include: ${validUnits.slice(0, 5).join(', ')}, ...`
        });
      }
    }
  });

  return { errors, warnings, all: [...errors, ...warnings] };
}

/**
 * Plate layout definitions (rows x columns).
 * @type {Object.<string, {rows: number, cols: number}>}
 */
const PLATE_LAYOUTS = {
  '12-well': { rows: 3, cols: 4, rowNames: ['A', 'B', 'C'], colNames: ['1', '2', '3', '4'] },
  '24-well': { rows: 4, cols: 6, rowNames: ['A', 'B', 'C', 'D'], colNames: ['1', '2', '3', '4', '5', '6'] },
  '96-well': { rows: 8, cols: 12, rowNames: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], colNames: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'] },
};

/**
 * Generate plate metadata JSON.
 *
 * @param {Object} config - Configuration object
 * @param {string} config.version - 'v0.4' or 'v0.5'
 * @param {string} config.plateType - '12-well', '24-well', or '96-well'
 * @param {Array<{row: number, col: number}>} config.selectedWells - Array of selected well positions
 * @param {number} config.numFOVs - Number of fields of view per well
 * @returns {string} Formatted JSON string
 */
export function generatePlateJSON({ version, plateType, selectedWells, numFOVs }) {
  const layout = PLATE_LAYOUTS[plateType];
  if (!layout) {
    throw new Error(`Unknown plate type: ${plateType}`);
  }

  const rows = layout.rowNames.map(name => ({ name }));
  const columns = layout.colNames.map(name => ({ name }));

  const wells = selectedWells.map(({ row, col }) => ({
    path: `${layout.rowNames[row]}/${layout.colNames[col]}`,
    rowIndex: row,
    columnIndex: col
  }));

  const plateDef = {
    columns,
    rows,
    wells,
    field_count: numFOVs,
    name: 'example_plate',
  };

  if (version === 'v0.4') {
    plateDef.version = '0.4';
  }

  const output = version === 'v0.5'
    ? {
        zarr_format: 3,
        node_type: 'group',
        attributes: {
          ome: {
            version: '0.5',
            plate: plateDef
          }
        }
      }
    : { plate: plateDef };

  return compactJSON(output);
}

/**
 * Generate well metadata JSON.
 *
 * @param {Object} config - Configuration object
 * @param {string} config.version - 'v0.4' or 'v0.5'
 * @param {number} config.numFOVs - Number of fields of view
 * @returns {string} Formatted JSON string
 */
export function generateWellJSON({ version, numFOVs }) {
  const images = Array.from({ length: numFOVs }, (_, i) => ({
    path: String(i)
  }));

  const wellDef = { images };

  if (version === 'v0.4') {
    wellDef.version = '0.4';
  }

  const output = version === 'v0.5'
    ? {
        zarr_format: 3,
        node_type: 'group',
        attributes: {
          ome: {
            version: '0.5',
            well: wellDef
          }
        }
      }
    : { well: wellDef };

  return compactJSON(output);
}

/**
 * Generate Python code for creating a plate using yaozarrs library.
 *
 * @param {Object} config - Configuration object
 * @param {Dimension[]} config.dimensions - Array of dimension objects
 * @param {string} config.version - 'v0.4' or 'v0.5'
 * @param {number} config.numLevels - Number of pyramid levels
 * @param {string} config.plateType - '12-well', '24-well', or '96-well'
 * @param {Array<{row: number, col: number}>} config.selectedWells - Array of selected well positions
 * @param {number} config.numFOVs - Number of fields of view per well
 * @returns {string} Python code string
 */
export function generatePlatePython({ dimensions, version, numLevels, plateType, selectedWells, numFOVs }) {
  const ver = version.replace('.', '');
  const layout = PLATE_LAYOUTS[plateType];

  // Generate the multiscale code (reuse from generatePython)
  const axesCode = dimensions.map(d => {
    const typeMap = {
      'space': 'SpaceAxis',
      'time': 'TimeAxis',
      'channel': 'ChannelAxis',
    };
    const axisClass = typeMap[d.type] || 'CustomAxis';
    const unit = d.unit ? `, unit="${d.unit}"` : '';
    return `    ${ver}.${axisClass}(name="${d.name}"${unit})`;
  }).join(',\n');

  const dimsCode = dimensions.map(d => {
    const parts = [`name="${d.name}"`];
    if (d.type) parts.push(`type="${d.type}"`);
    if (d.unit) parts.push(`unit="${d.unit}"`);
    if (d.scale !== 1) parts.push(`scale=${d.scale}`);
    if (d.translation !== 0) parts.push(`translation=${d.translation}`);
    if (d.scaleFactor !== 1) parts.push(`scale_factor=${d.scaleFactor}`);
    return `    DimSpec(${parts.join(', ')})`;
  }).join(',\n');

  // Generate rows and columns
  const rowsCode = layout.rowNames.map(name => `${ver}.Row(name="${name}")`).join(', ');
  const colsCode = layout.colNames.map(name => `${ver}.Column(name="${name}")`).join(', ');

  // Generate wells
  const wellsCode = selectedWells.map(({ row, col }) => {
    const rowName = layout.rowNames[row];
    const colName = layout.colNames[col];
    return `    ${ver}.PlateWell(path="${rowName}/${colName}", rowIndex=${row}, columnIndex=${col})`;
  }).join(',\n');

  // Generate fields of view
  const fovsCode = Array.from({ length: numFOVs }, (_, i) =>
    `    ${ver}.FieldOfView(path="${i}")`
  ).join(',\n');

  return `from yaozarrs import ${ver}, DimSpec

# ###############################################
# Define image dimensions (same for all FOVs)
# ###############################################

dims = [
${dimsCode}
]

multiscale = ${ver}.Multiscale.from_dims(
    dims,
    name="example_image",
    n_levels=${numLevels}
)

# ###############################################
# Define plate structure
# ###############################################

rows = [${rowsCode}]
columns = [${colsCode}]

wells_list = [
${wellsCode}
]

plate_def = ${ver}.PlateDef(
    rows=rows,
    columns=columns,
    wells=wells_list,
    field_count=${numFOVs},
    name="example_plate"${version === 'v0.4' ? ',\n    version="0.4"' : ''}
)

# ###############################################
# Define well with fields of view
# ###############################################

well_images = [
${fovsCode}
]

well_def = ${ver}.WellDef(
    images=well_images${version === 'v0.4' ? ',\n    version="0.4"' : ''}
)

# ###############################################
# Print JSON metadata
# ###############################################

# Plate metadata (root level)
${version === 'v0.5' ? `plate = ${ver}.Plate(version="0.5", plate=plate_def)
print(plate.model_dump_json(indent=2))` : `print(plate_def.model_dump_json(indent=2))`}

# Well metadata (A/1/zarr.json)
${version === 'v0.5' ? `well = ${ver}.Well(version="0.5", well=well_def)
print(well.model_dump_json(indent=2))` : `print(well_def.model_dump_json(indent=2))`}

# Image metadata (A/1/0/zarr.json) - same as single image
image = ${ver}.Image(multiscales=[multiscale])
print(image.model_dump_json(indent=2))`
;
}
