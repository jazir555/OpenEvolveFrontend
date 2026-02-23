"use strict";
/**
 * OpenEvolve BubbleLabs Plugin - Main Exports
 *
 * This file exports all public APIs, components, hooks, and utilities
 * for the OpenEvolve plugin, following the same pattern as other BubbleLabs plugins.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OPENEVOLVE_PLUGIN_INFO = exports.OpenEvolveConfigPanel = exports.openevolvePlugin = void 0;
// Export core types
__exportStar(require("./types/plugin-types"), exports);
// Export extended types
__exportStar(require("./types/extended-plugin-types"), exports);
// Export utils
__exportStar(require("./utils/createOpenEvolvePlugin"), exports);
// Export the global plugin instance
var createOpenEvolvePlugin_1 = require("./utils/createOpenEvolvePlugin");
Object.defineProperty(exports, "openevolvePlugin", { enumerable: true, get: function () { return createOpenEvolvePlugin_1.openevolvePlugin; } });
// Export React components
var OpenEvolveConfigPanel_1 = require("./components/OpenEvolveConfigPanel");
Object.defineProperty(exports, "OpenEvolveConfigPanel", { enumerable: true, get: function () { return OpenEvolveConfigPanel_1.OpenEvolveConfigPanel; } });
// export * from './components/OpenEvolveExecutionPanel';
// Export React hooks (will be implemented)
// export * from './hooks/useOpenEvolveConfig';
// export * from './hooks/useOpenEvolveState';
// export * from './hooks/useOpenEvolveExecution';
// Export services (will be implemented)
// export * from './services/OpenEvolveClient';
// export * from './services/OpenEvolveService';
/**
 * Plugin Information
 */
exports.OPENEVOLVE_PLUGIN_INFO = {
    name: 'OpenEvolve BubbleLabs Plugin',
    version: '2.0.0',
    description: 'Comprehensive OpenEvolve system integration for BubbleLabs with extended features',
    author: 'OpenEvolve Team',
    license: 'MIT',
    repository: 'https://github.com/openevolve/openevolve-bubblelab-plugin',
    documentation: 'https://openevolve.github.io/openevolve-bubblelab-plugin',
};
//# sourceMappingURL=index.js.map