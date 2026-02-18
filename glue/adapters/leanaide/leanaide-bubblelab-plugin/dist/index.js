/**
 * Main Entry Point for LeanAide BubbleLab Integration
 *
 * This file provides the main exports for the complete LeanAide autoformalization
 * system with predictive analytics integration.
 */
// Main integration components
export { default as LeanAideBubbleLabIntegration, LeanAideBubbleLabIntegrationProps } from './BubbleLabIntegration';
export { BubbleLabLeanAideIntegrationLazy } from './BubbleLabIntegration';
// Core autoformalization components
export { LeanAideAutoformalizationEngine, AutoformalizationResult, AutoformalizationStrategy, create_leanaide_autoformalization_engine, autoformalize_with_mdap_maker } from './integration/autoformalizationAnalytics';
// Enhanced verification components
export { EnhancedLeanAideVerification } from './integration/autoformalizationAnalytics';
// Analytics components
export { AnalyticsDashboard } from './integration/autoformalizationAnalytics';
// Knowledge graph components
export { KnowledgeGraphIntegration } from './integration/autoformalizationAnalytics';
// Plugin system
export { LeanAidePlugin, pluginRegistry, PluginManager, PluginManagerProvider, usePluginManager } from './PluginSystem';
// Analytics hooks and services
export { useAutoformalizationAnalytics } from './integration/autoformalizationAnalytics';
// Services
export { initializeLeanAideClient, initializeRagbitsClient, translateTheorem, translateDefinition, verifySolution, elaborateCode, mathQuery, searchKnowledge, ingestArtifact, isLeanAideAvailable, isRagbitsAvailable } from './services';
// Configuration
export { DEFAULT_ANALYTICS_CONFIG } from './integration/autoformalizationAnalytics';
// Plugin registration
export { registerBubbleLabIntegration } from './BubbleLabIntegration';
//# sourceMappingURL=index.js.map