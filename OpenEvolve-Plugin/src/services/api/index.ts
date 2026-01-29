/**
 * OpenEvolve API Client - Central Export
 *
 * This file exports all API-related functionality for the OpenEvolve integration.
 */

export { apiClient, ApiClient } from './client';
export type { ApiResponse } from './client';

export { api } from './endpoints';
export {
  authApi,
  userApi,
  evolutionApi,
  adversarialApi,
  analyticsApi,
  monitoringApi,
  contentApi,
  versionApi,
  collaborationApi,
  commentsApi,
  configApi,
  workflowApi,
  filesApi,
  leanaideApi,
} from './endpoints';

export {
  WebSocketClient,
  createEvolutionWebSocket,
  createAdversarialWebSocket,
  createCollaborationWebSocket,
  createMonitoringWebSocket,
} from './websocket';
export type {
  WebSocketMessage,
  WebSocketMessageType,
  ConnectionState,
  WebSocketConfig,
  WebSocketHandlers,
} from './websocket';

// Export OpenEvolve comprehensive API service
export { openEvolveAPI, OpenEvolveAPIClass } from './OpenEvolveAPI';
export type {
  // Evolution types
  EvolutionConfig,
  EvolutionRun,
  EvolutionCreateRequest,
  EvolutionUpdateRequest,
  // Adversarial types
  AdversarialConfig,
  AdversarialRun,
  AdversarialCreateRequest,
  AdversarialUpdateRequest,
  // Knowledge Base types
  KnowledgeEntry,
  KnowledgeCategory,
  KnowledgeCreateRequest,
  KnowledgeUpdateRequest,
  KnowledgeQueryParams,
  KnowledgeStats,
  // Workflow types
  WorkflowNode,
  WorkflowEdge,
  WorkflowDefinition,
  WorkflowCreateRequest,
  WorkflowUpdateRequest,
  WorkflowInstance,
  // Analytics types
  WorkflowPerformance,
  TeamPerformance,
  GauntletPerformance,
  SolutionQuality,
  AnalyticsQueryParams,
  // Decomposition types
  DecompositionProblem,
  SubProblem,
  DecompositionRequest,
} from './OpenEvolveAPI';

// Export OpenEvolve React hooks
export {
  // Evolution hooks
  useEvolutionRuns,
  useEvolutionRun,
  useCreateEvolutionRun,
  useEvolutionConfig,
  // Adversarial hooks
  useAdversarialRuns,
  useAdversarialRun,
  useCreateAdversarialRun,
  useAdversarialConfig,
  // Knowledge Base hooks
  useKnowledgeEntries,
  useKnowledgeCategories,
  useKnowledgeStats,
  useCreateKnowledgeEntry,
  // Workflow hooks
  useWorkflows,
  useWorkflow,
  useWorkflowInstances,
  useWorkflowTemplates,
  // Analytics hooks
  useWorkflowPerformance,
  useTeamPerformance,
  useGauntletPerformance,
  useSolutionQuality,
  useAnalyticsOverview,
  // Decomposition hooks
  useDecompositionProblems,
  useSubProblems,
  // System hooks
  useHealthStatus,
  useSystemStatus,
} from './OpenEvolveAPIHooks';

// Export graceful error handling utilities
export { gracefulErrorHandler, withGracefulErrorHandling, useGracefulErrorHandler } from '../../utils/gracefulErrorHandler';
export type { ErrorHandlingOptions, ErrorHandlingResult, ErrorHandlingStrategy, ErrorContext } from '../../utils/gracefulErrorHandler';
