/**
 * OpenEvolve BubbleLabs Plugin - Main Exports
 *
 * This file exports all public APIs, components, hooks, and utilities
 * for the OpenEvolve plugin, following the same pattern as other BubbleLabs plugins.
 */
export * from './types/plugin-types';
export * from './types/extended-plugin-types';
export * from './utils/createOpenEvolvePlugin';
export { openevolvePlugin } from './utils/createOpenEvolvePlugin';
export { OpenEvolveConfigPanel } from './components/OpenEvolveConfigPanel';
/**
 * Plugin Information
 */
export declare const OPENEVOLVE_PLUGIN_INFO: {
    name: string;
    version: string;
    description: string;
    author: string;
    license: string;
    repository: string;
    documentation: string;
};
//# sourceMappingURL=index.d.ts.map