/**
 * Main Entry Point for LeanAide BubbleLab Integration
 *
 * This file provides the main exports for the complete LeanAide autoformalization
 * system with predictive analytics integration.
 */
export { default as LeanAideBubbleLabIntegration, LeanAideBubbleLabIntegrationProps } from './BubbleLabIntegration';
export { BubbleLabLeanAideIntegrationLazy } from './BubbleLabIntegration';
export { LeanAideAutoformalizationEngine, AutoformalizationResult, AutoformalizationStrategy, create_leanaide_autoformalization_engine, autoformalize_with_mdap_maker } from './integration/autoformalizationAnalytics';
export { EnhancedLeanAideVerification, EnhancedLeanAideVerificationProps } from './integration/autoformalizationAnalytics';
export { AnalyticsDashboard, AnalyticsDashboardProps } from './integration/autoformalizationAnalytics';
export { KnowledgeGraphIntegration, KnowledgeGraphIntegrationProps } from './integration/autoformalizationAnalytics';
export { LeanAidePlugin, LeanAidePluginInterface, LeanAidePluginConfig, LeanAidePluginLifecycle, pluginRegistry, PluginManager, PluginManagerProvider, usePluginManager } from './PluginSystem';
export { useAutoformalizationAnalytics } from './integration/autoformalizationAnalytics';
export type { AutoformalizationMetrics, AutoformalizationEvent, AutoformalizationConfig, AutoformalizationStrategy as AutoformalizationStrategyType, AutoformalizationResult as AutoformalizationResultType } from './integration/autoformalizationAnalytics';
export { initializeLeanAideClient, initializeRagbitsClient, translateTheorem, translateDefinition, verifySolution, elaborateCode, mathQuery, searchKnowledge, ingestArtifact, isLeanAideAvailable, isRagbitsAvailable } from './services';
export { DEFAULT_ANALYTICS_CONFIG } from './integration/autoformalizationAnalytics';
export { registerBubbleLabIntegration } from './BubbleLabIntegration';
