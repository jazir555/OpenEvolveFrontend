import { OpenEvolvePlugin, OpenEvolvePluginState } from '../types/plugin-types';
/**
 * OpenEvolve Plugin Factory Function
 * Creates a new OpenEvolve plugin instance with full functionality
 */
export declare function createOpenEvolvePlugin(initialConfig?: Partial<OpenEvolvePluginState>): OpenEvolvePlugin;
/**
 * Global OpenEvolve Plugin Instance
 * Singleton instance that can be imported and used throughout the application
 */
export declare const openevolvePlugin: OpenEvolvePlugin;
export type { OpenEvolvePlugin, OpenEvolvePluginState, OpenEvolveExecutionOptions, OpenEvolveExecutionResult, OpenEvolveExecutionStatistics, } from '../types/plugin-types';
