#!/usr/bin/env node
/**
 * Node.js test runner for OME explorer generation functions.
 *
 * This script imports the pure generation functions from ome_generator.js
 * and runs them with configurations provided via command line arguments.
 *
 * Usage:
 *   node tests/ome_explorer_runner.js '<json_config>'
 *
 * Input (JSON config):
 *   {
 *     "dimensions": [
 *       {"name": "y", "type": "space", "unit": "micrometer", "scale": 0.5, "translation": 0, "scaleFactor": 2},
 *       {"name": "x", "type": "space", "unit": "micrometer", "scale": 0.5, "translation": 0, "scaleFactor": 2}
 *     ],
 *     "version": "v0.4",
 *     "numLevels": 3
 *   }
 *
 * Output (JSON):
 *   {
 *     "json": "<generated OME-NGFF JSON string>",
 *     "python": "<generated Python code string>"
 *   }
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Get the directory of this script
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import the generator module (units are loaded automatically via ome_units.js)
const generatorPath = join(__dirname, '..', 'docs', 'javascripts', 'ome_generator.js');
const {
  generateJSON,
  generatePython,
  generatePlateJSON,
  generateWellJSON,
  generatePlatePython
} = await import(generatorPath);

// Parse config from command line
const configArg = process.argv[2];
if (!configArg) {
  console.error('Usage: node tests/ome_explorer_runner.js \'<json_config>\'');
  process.exit(1);
}

let config;
try {
  config = JSON.parse(configArg);
} catch (e) {
  console.error(`Error parsing config JSON: ${e.message}`);
  process.exit(1);
}

// Generate outputs based on config type
let output;
if (config.isPlate) {
  // For plate configs, generate plate, well, and Python code
  output = {
    plateJson: generatePlateJSON(config),
    wellJson: generateWellJSON(config),
    python: generatePlatePython(config),
  };
} else {
  // Standard image config
  output = {
    json: generateJSON(config),
    python: generatePython(config),
  };
}

// Output as JSON
console.log(JSON.stringify(output));
