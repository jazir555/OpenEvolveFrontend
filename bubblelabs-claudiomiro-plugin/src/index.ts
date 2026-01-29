// BubbleLabs ClaudieMiro Plugin - Main Export
// Standalone plugin for autonomous development workflows

// Export types
export type {
  ClaudieMiroPluginConfig,
  ClaudieMiroPluginState,
  ClaudieMiroDevelopmentResult,
  ClaudieMiroCritiqueResult,
  ClaudieMiroTestResult,
  ClaudieMiroValidationResult,
  ClaudieMiroPluginContext,
  ClaudieMiroPluginMethods,
  ClaudieMiroPlugin,
  ClaudieMiroPluginProps,
  ClaudieMiroConfigPanelProps,
  ClaudieMiroDevelopmentPanelProps,
  ClaudieMiroPhaseMonitorProps,
  ClaudieMiroStatusIndicatorProps,
  ClaudieMiroWorkflowSelectorProps,
  ClaudieMiroWorkflow,
  ClaudieMiroPhase
} from './types/plugin-types';

export {
  CLAUDIEMIRO_WORKFLOWS,
  CLAUDIEMIRO_PHASES,
  DEFAULT_CLAUDIEMIRO_CONFIG
} from './types/plugin-types';

// Export components (will be implemented)
export { ClaudieMiroConfigPanel } from './components/ClaudieMiroConfigPanel';
export { ClaudieMiroDevelopmentPanel } from './components/ClaudieMiroDevelopmentPanel';
export { ClaudieMiroPhaseMonitor } from './components/ClaudieMiroPhaseMonitor';
export { ClaudieMiroStatusIndicator } from './components/ClaudieMiroStatusIndicator';
export { ClaudieMiroWorkflowSelector } from './components/ClaudieMiroWorkflowSelector';

// Export hooks (will be implemented)
export { useClaudieMiroConfig } from './hooks/useClaudieMiroConfig';
export { useClaudieMiroState } from './hooks/useClaudieMiroState';
export { useClaudieMiroDevelopment } from './hooks/useClaudieMiroDevelopment';
export { useClaudieMiroPhase } from './hooks/useClaudieMiroPhase';

// Export services (will be implemented)
export { ClaudieMiroClient } from './services/ClaudieMiroClient';
export { ClaudieMiroService } from './services/ClaudieMiroService';

// Export utilities (will be implemented)
export { createClaudieMiroPlugin } from './utils/createClaudieMiroPlugin';
export { useClaudieMiroPlugin } from './hooks/useClaudieMiroPlugin';

// Export the main plugin factory
import { createClaudieMiroPlugin } from './utils/createClaudieMiroPlugin';

/**
 * Create a new ClaudieMiro plugin instance
 * @param config Optional initial configuration
 * @returns ClaudieMiroPlugin instance
 */
export function createPlugin(config?: Partial<ClaudieMiroPluginConfig>): ClaudieMiroPlugin {
  return createClaudieMiroPlugin(config);
}

/**
 * Default plugin instance
 */
export const ClaudieMiroPlugin = createClaudieMiroPlugin();

export default ClaudieMiroPlugin;