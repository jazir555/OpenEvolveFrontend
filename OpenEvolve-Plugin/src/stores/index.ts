/**
 * Stores module - Central export point for all store types
 */

// Export types from auth store
export type { User } from './authStore';

// Export types from workflow store
export type { WorkflowExecution } from './workflowStore';

// Export types from analytics store
export type { AnalyticsData, PerformanceAnalytics } from './analyticsStore';

// Export types from knowledge store
export type { KnowledgeArtifact } from './knowledgeStore';

// Export types from evolution store
export type { AdversarialTest } from './evolutionStore';

// Export types from leanaide store
export type { LeanCodeOutput, VerificationResult } from './leanaideStore';

// Export types from settings store
export type {
  SettingsScope,
  GenerationSettings,
  EvolutionSettings,
  ScopedSettings,
  ProviderSettings,
  ParameterSettings,
} from './settingsStore';

// Re-export all stores
export { useAuthStore } from './authStore';
export { useWorkflowStore } from './workflowStore';
export { useAnalyticsStore } from './analyticsStore';
export { useKnowledgeStore } from './knowledgeStore';
export { useEvolutionStore } from './evolutionStore';
export { useLeanAideStore } from './leanaideStore';
export { useSettingsStore } from './settingsStore';
