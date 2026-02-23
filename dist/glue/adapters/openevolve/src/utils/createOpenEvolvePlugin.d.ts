/**
 * OpenEvolve BubbleLabs Plugin Factory
 *
 * This file implements the OpenEvolve plugin factory with comprehensive state management,
 * following the same pattern as other BubbleLabs plugins (LeanAIDE, ClaudieMiro, Datapizza, ROMA).
 *
 * Features:
 * - Singleton pattern with global instance management
 * - Zustand store for state management
 * - Complete plugin methods implementation
 * - MDAP/MAKER auto-selection logic
 * - Error handling and status tracking
 * - Evolution, Adversarial, and Decomposition functionality
 */
import { OpenEvolvePlugin, OpenEvolvePluginState, OpenEvolveExecutionOptions, OpenEvolveExecutionResult, OpenEvolveExecutionStatistics, DEFAULT_OPENEVOLVE_CONFIG, OPENEVOLVE_PLUGIN_CONSTANTS } from '../types/plugin-types';
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
export type { OpenEvolvePlugin, OpenEvolvePluginState, OpenEvolveExecutionOptions, OpenEvolveExecutionResult, OpenEvolveExecutionStatistics, };
export { DEFAULT_OPENEVOLVE_CONFIG, OPENEVOLVE_PLUGIN_CONSTANTS };
//# sourceMappingURL=createOpenEvolvePlugin.d.ts.map