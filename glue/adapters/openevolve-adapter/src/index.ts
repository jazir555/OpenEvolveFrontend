/**
 * OpenEvolve Adapter Public API
 *
 * This file exports all public components of the OpenEvolve main orchestration adapter.
 */

// Main adapter
export {
  OpenEvolveAdapter,
  createOpenEvolveAdapter,
  OpenEvolveAdapterConfig,
} from './adapter';

// Type exports from adapter
export type {
  ModelConfig,
  Team,
  Gauntlet,
  GauntletRoundRule,
  SubProblem,
  SolutionAttempt,
  CritiqueReport,
  VerificationReport,
  WorkflowDefinition,
  WorkflowState,
  KnowledgeArtifact,
  IntegrationHealth,
  LogContext,
} from './adapter';

// Integration coordinator
export {
  IntegrationCoordinator,
  createIntegrationCoordinator,
} from './integration-coordinator';

export type {
  AdapterEndpoint,
  CoordinationRequest,
  CoordinationResult,
  CoordinationPlan,
} from './integration-coordinator';

// Workflow orchestrator
export {
  WorkflowOrchestrator,
  createWorkflowOrchestrator,
} from './workflow-orchestrator';

export type {
  WorkflowExecutionRequest,
  WorkflowProgressUpdate,
  WorkflowExecutionResult,
  WorkflowError,
  StageDefinition,
} from './workflow-orchestrator';

// Knowledge aggregator
export {
  KnowledgeAggregator,
  createKnowledgeAggregator,
} from './knowledge-aggregator';

export type {
  KnowledgeQuery,
  KnowledgeResult,
  KnowledgeFusionResult,
  KnowledgeExtractionRequest,
  KnowledgeGraphEdge,
  KnowledgeGraphNode,
} from './knowledge-aggregator';

// Utility exports
export { StructuredLogger } from './adapter';
