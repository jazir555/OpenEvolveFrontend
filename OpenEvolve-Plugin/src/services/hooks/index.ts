/**
 * OpenEvolve Custom Hooks - Central Export
 *
 * This file exports all custom React hooks for the OpenEvolve integration.
 */

export {
  useApi,
  useAuth,
  useEvolution,
  useEvolutions,
  useAdversarialTest,
  useAdversarialTests,
  useAnalytics,
  useContent,
  useContentList,
  useKnowledgeGraph,
  useMonitoring,
  useConfig,
  useLeanAide,
} from './useApi';

export {
  useWebSocket,
  useEvolutionWebSocket,
  useAdversarialWebSocket,
  useCollaborationWebSocket,
  useMonitoringWebSocket,
} from './useWebSocket';

export {
  useRealtimeEvolution,
  useRealtimeAdversarial,
  useRealtimeMonitoring,
  useAutoRefresh,
  useRealtime,
} from './useRealtime';

export {
  useWorkflows,
  useWorkflow,
  useWorkflowConfig,
  useWorkflowModels,
  useIntegratedWorkflow,
} from './useWorkflows';

export {
  useKnowledge,
  useArtifact,
  useArtifactVersions,
  useArtifactDiff,
  useKnowledgeSearch,
  useArtifactComments,
  useCollaboration,
} from './useKnowledge';
