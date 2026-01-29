// @ts-nocheck
/**
 * Core Infrastructure - Unified Entry Point
 * Merges all core types, utilities, and plugin definitions from all 3 plugins
 */

// Plugin Definition (from embedded P3)
export { default as PluginDefinition } from './plugin/PluginDefinition';

// Types (merged from P1 and P2)
export * from './types/plugin';
export * from './types/enhanced-plugin-types';
export * from './types/extended-plugin-types';
export * from './types/nodes';

// Utilities (merged from P1 and P2)
export * from './utils/helpers';
export * from './utils/validators';
