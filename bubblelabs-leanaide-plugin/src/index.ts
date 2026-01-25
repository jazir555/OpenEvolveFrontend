// BubbleLabs LeanAIDE Plugin - Main Export
// Standalone plugin for mathematical autoformalization and verification

// Export types
export type {
  LeanAIDEPluginConfig,
  LeanAIDEPluginState,
  LeanAIDEAutoformalizationResult,
  LeanAIDEVerificationResult,
  LeanAIDEPluginContext,
  LeanAIDEPluginMethods,
  LeanAIDEPlugin,
  LeanAIDEPluginProps,
  LeanAIDEConfigPanelProps,
  LeanAIDEAutoformalizationPanelProps,
  LeanAIDEVerificationPanelProps,
  LeanAIDEStrategySelectorProps,
  LeanAIDEStatusIndicatorProps,
  LeanAIDEStrategy,
  LeanAIDEMathematicalDomain
} from './types/plugin-types';

export {
  LEANAIDE_STRATEGIES,
  MATHEMATICAL_DOMAINS,
  DEFAULT_LEANAIDE_CONFIG
} from './types/plugin-types';

// Export components
export { LeanAIDEConfigPanel } from './components/LeanAIDEConfigPanel';
export { LeanAIDEAutoformalizationPanel } from './components/LeanAIDEAutoformalizationPanel';
export { LeanAIDEVerificationPanel } from './components/LeanAIDEVerificationPanel';
export { LeanAIDEStrategySelector } from './components/LeanAIDEStrategySelector';
export { LeanAIDEStatusIndicator } from './components/LeanAIDEStatusIndicator';

// Export hooks
export { useLeanAIDEConfig } from './hooks/useLeanAIDEConfig';
export { useLeanAIDEState } from './hooks/useLeanAIDEState';
export { useLeanAIDEAutoformalization } from './hooks/useLeanAIDEAutoformalization';
export { useLeanAIDEVerification } from './hooks/useLeanAIDEVerification';

// Export services
export { LeanAIDEClient } from './services/LeanAIDEClient';
export { LeanAIDEService } from './services/LeanAIDEService';

// Export utilities
export { createLeanAIDEPlugin } from './utils/createLeanAIDEPlugin';
export { useLeanAIDEPlugin } from './hooks/useLeanAIDEPlugin';

// Export the main plugin factory
import { createLeanAIDEPlugin } from './utils/createLeanAIDEPlugin';

/**
 * Create a new LeanAIDE plugin instance
 * @param config Optional initial configuration
 * @returns LeanAIDEPlugin instance
 */
export function createPlugin(config?: Partial<LeanAIDEPluginConfig>): LeanAIDEPlugin {
  return createLeanAIDEPlugin(config);
}

/**
 * Default plugin instance
 */
export const LeanAIDEPlugin = createLeanAIDEPlugin();

export default LeanAIDEPlugin;