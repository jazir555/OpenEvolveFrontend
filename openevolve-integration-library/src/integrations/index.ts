/**
 * OpenEvolve Integration Adapters - Main Export
 *
 * Exports all integration adapters with clean, typed interfaces
 *
 * @example
 * ```typescript
 * import { LeanAideIntegration, EvolutionIntegration } from '@openevolve/integration-library';
 *
 * const leanaide = new LeanAideIntegration(client);
 * const proof = await leanaide.generateProof('theorem', 'strategy');
 * ```
 */

// Export base adapter
export { BaseIntegrationAdapter } from './base';

// Export all integration adapters
export {
  LeanAideIntegration,
  EvolutionIntegration,
  KnowledgeIntegration,
  MakerIntegration,
  CrewAIIntegration,
  DecompositionIntegration,
  VerificationIntegration,
  AssemblyIntegration,
  SolutionIntegration,
} from './all-integrations';

// Re-export common types
export type {
  ValidationResult,
  ParameterSchema,
  ProgressUpdate,
  ExecutionOptions,
} from '../api/types';

// Re-export integration types
export type {
  LeanAideInputs,
  LeanAideResult,
  EvolutionInputs,
  EvolutionResult,
  KnowledgeInputs,
  KnowledgeResult,
  MakerInputs,
  MakerResult,
  CrewAIInputs,
  CrewAIResult,
  DecompositionInputs,
  DecompositionResult,
  VerificationInputs,
  VerificationResult,
  AssemblyInputs,
  AssemblyResult,
  SolutionInputs,
  SolutionResult,
} from './all-integrations';
