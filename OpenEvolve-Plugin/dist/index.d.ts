import { EvolutionConfig, AdversarialConfig, DecompositionConfig, IntegrationConfig, OpenEvolveNodeConfig, OpenEvolvePlugin } from './types';
/**
 * OpenEvolve BubbleLab Plugin - Main Export
 *
 * Comprehensive plugin for integrating OpenEvolve workflow capabilities
 * with BubbleLab's visual programming interface.
 *
 * @package @openevolve/bubblelab-plugin
 * @version 1.0.0
 * @license MIT
 * @author OpenEvolve Team
 */
export * from './types/plugin-types';
export * from './types/enhanced-plugin-types';
export * from './types/extended-plugin-types';
export * from './types';
export type { OpenEvolveNodeData, OpenEvolveNodeConfig, EvolutionNodeData, EvolutionConfig, EvolutionResult, EvolutionStrategy, AdversarialNodeData, AdversarialConfig, AdversarialResult, AttackStrategy, DecompositionNodeData, DecompositionConfig, DecompositionResult, DecompositionStrategy, IntegrationConfig, IntegrationResult, KnowledgeEngineConfig, LeanAIDEConfig, HephaestusConfig, OpenEvolvePlugin, PluginContext, PluginState, PluginActions, } from './types';
export * from './nodes';
export { NodeRegistry, registerNodes, createNodeFromConfig as createNodeFromRegistryConfig, type NodeClass, type NodeMetadata, type NodeCreationConfig, type ValidationResult, type RegistrationOptions, type RegistryStats, } from './nodes/registry';
export * from './components';
export { EnhancedOpenEvolveConfigPanel, OpenEvolveConfigPanel, } from './components';
export { useEnhancedOpenEvolveConfig, } from './hooks/useEnhancedOpenEvolveConfig';
export * from './utils';
export { createOpenEvolvePlugin, } from './utils/createOpenEvolvePlugin';
export { createEnhancedOpenEvolvePlugin, getEnhancedOpenEvolvePlugin, resetEnhancedOpenEvolvePlugin, } from './utils/createEnhancedOpenEvolvePlugin';
export declare const PLUGIN_NAME = "@openevolve/bubblelab-plugin";
export declare const PLUGIN_VERSION = "1.0.0";
export declare const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig;
export declare const DEFAULT_ADVERSARIAL_CONFIG: AdversarialConfig;
export declare const DEFAULT_DECOMPOSITION_CONFIG: DecompositionConfig;
export declare const DEFAULT_INTEGRATION_CONFIG: IntegrationConfig;
/**
 * Create and initialize the OpenEvolve BubbleLab plugin
 * @param config Optional plugin configuration
 * @returns Initialized plugin instance
 */
export declare function createPlugin(config?: Partial<OpenEvolveNodeConfig>): OpenEvolvePlugin;
/**
 * Get the singleton plugin instance
 * @returns Plugin instance or throws if not initialized
 */
export declare function getPlugin(): OpenEvolvePlugin;
/**
 * Reset the singleton plugin instance
 */
export declare function resetPlugin(): void;
declare global {
    interface Window {
        __openevolve_plugin__?: OpenEvolvePlugin;
    }
}
