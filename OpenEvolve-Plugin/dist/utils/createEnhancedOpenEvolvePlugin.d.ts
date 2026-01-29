import { EnhancedOpenEvolvePluginState, EnhancedOpenEvolvePlugin } from '../types/enhanced-plugin-types';
/**
 * Enhanced OpenEvolve Plugin Factory
 * Creates a plugin with extended performance, security, monitoring, integration, and error handling capabilities
 */
export declare function createEnhancedOpenEvolvePlugin(initialConfig?: Partial<EnhancedOpenEvolvePluginState>): EnhancedOpenEvolvePlugin;
/**
 * Get or create the singleton enhanced plugin instance
 */
export declare function getEnhancedOpenEvolvePlugin(initialConfig?: Partial<EnhancedOpenEvolvePluginState>): EnhancedOpenEvolvePlugin;
/**
 * Reset the singleton enhanced plugin instance
 */
export declare function resetEnhancedOpenEvolvePlugin(): void;
