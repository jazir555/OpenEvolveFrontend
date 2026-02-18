/**
 * OpenEvolve BubbleLabs Plugin - Main Exports
 * 
 * This file exports all public APIs, components, hooks, and utilities
 * for the OpenEvolve plugin, following the same pattern as other BubbleLabs plugins.
 */

// Export core types
export * from './types/plugin-types';

// Export extended types
export * from './types/extended-plugin-types';

// Export utils
export * from './utils/createOpenEvolvePlugin';

// Export the global plugin instance
export { openevolvePlugin } from './utils/createOpenEvolvePlugin';

// Export React components
export { default as OpenEvolveConfigPanel } from './components/OpenEvolveConfigPanel';
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
export const OPENEVOLVE_PLUGIN_INFO = {
  name: 'OpenEvolve BubbleLabs Plugin',
  version: '2.0.0',
  description: 'Comprehensive OpenEvolve system integration for BubbleLabs with extended features',
  author: 'OpenEvolve Team',
  license: 'MIT',
  repository: 'https://github.com/openevolve/openevolve-bubblelab-plugin',
  documentation: 'https://openevolve.github.io/openevolve-bubblelab-plugin',
};
